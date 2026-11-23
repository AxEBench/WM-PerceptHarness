"""Strict natural event contracts; times always refer to the source video."""
from typing import Annotated, Literal
import math
from pydantic import Field, StrictStr, StrictBool, field_validator, model_validator
from ...cv.contracts import StrictModel, Timestamp, PositiveTimestamp, Fraction

Text = Annotated[StrictStr, Field(min_length=1, max_length=4000)]
Label = Annotated[StrictStr, Field(min_length=1, max_length=256)]


class CrossFieldValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues[:16]
        super().__init__('; '.join(self.issues))


class Event(StrictModel):
    start: Timestamp
    end: PositiveTimestamp
    description: Text

    @field_validator('description')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('description must not be blank')
        return value

    @model_validator(mode='after')
    def interval(self):
        if self.end <= self.start:
            raise ValueError('event must have positive duration')
        return self


class GroundedEvents(StrictModel):
    duration: PositiveTimestamp
    events: Annotated[tuple[Event, ...], Field(max_length=2000)]

    @model_validator(mode='after')
    def bounds(self):
        if any(event.end > self.duration for event in self.events):
            raise ValueError('event outside original video')
        return self


class ObservationAssessment(StrictModel):
    category: Literal['observable_defect', 'visual_uncertainty', 'pipeline_error', 'no_source_limitation']
    start: Timestamp
    end: PositiveTimestamp
    reason: Annotated[StrictStr, Field(min_length=1, max_length=4000, pattern=r'\S')]
    visual_times: Annotated[tuple[Timestamp, ...], Field(min_length=1, max_length=8)]
    event_index: Annotated[int, Field(ge=0, strict=True)]
    candidate_quote: Annotated[StrictStr, Field(min_length=1, max_length=4000, pattern=r'\S')]

    @model_validator(mode='after')
    def interval(self):
        if self.end <= self.start:
            raise ValueError('assessment requires a positive window')
        return self


class ConcernResponse(StrictModel):
    concern_id: Label
    action: Literal['changed', 'retained', 'unresolved']
    rationale: Annotated[StrictStr, Field(min_length=1, max_length=4000, pattern=r'\S')]
    visual_times: Annotated[tuple[Timestamp, ...], Field(min_length=1, max_length=8)]
    assessment: ObservationAssessment | None = None


class ReviewedRevision(GroundedEvents):
    responses: Annotated[tuple[ConcernResponse, ...], Field(max_length=4096)]


class ConcernVerdict(StrictModel):
    concern_id: Label
    resolved: StrictBool
    reason: Annotated[StrictStr, Field(min_length=1, max_length=4000, pattern=r'\S')]
    assessment: ObservationAssessment | None = None


class SpatialTarget(StrictModel):
    description: Label
    reference_time: Timestamp
    bbox_xyxy: tuple[Fraction, Fraction, Fraction, Fraction]

    @model_validator(mode='after')
    def valid(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        if not self.description.strip() or x1 >= x2 or y1 >= y2:
            raise ValueError('target needs a description and positive normalized box')
        return self


class EventUncertainty(StrictModel):
    kind: Literal['object_identity', 'object_location', 'occlusion', 'trajectory',
                  'event_boundary', 'description', 'missing_event', 'camera_motion']
    reason: Text
    start: Timestamp
    end: PositiveTimestamp
    target: SpatialTarget | None

    @model_validator(mode='after')
    def valid(self):
        if self.end <= self.start or not self.reason.strip():
            raise ValueError('invalid review concern')
        return self


class Concern(EventUncertainty):
    event_indices: Annotated[tuple[Annotated[int, Field(ge=0, strict=True)], ...], Field(max_length=2000)]


class RevisionVerification(StrictModel):
    verdicts: Annotated[tuple[ConcernVerdict, ...], Field(max_length=4096)]
    new_issues: Annotated[tuple[Concern, ...], Field(max_length=32)]


class SamTargetPlan(StrictModel):
    target: SpatialTarget | None
    reason: Annotated[StrictStr, Field(min_length=1, max_length=4000, pattern=r'\S')]


class SamPromptReview(StrictModel):
    accepted: StrictBool
    reason: Text
    correction: SpatialTarget | None

    @model_validator(mode='after')
    def valid(self):
        if not self.reason.strip() or (self.accepted and self.correction is not None):
            raise ValueError('accepted segmentation cannot request correction')
        return self


class EvidenceReview(StrictModel):
    motion_evidence: Annotated[StrictStr, Field(min_length=1, max_length=800, pattern=r"\S")]
    concerns: Annotated[tuple[Concern, ...], Field(max_length=32)]
    unresolved: Annotated[tuple[Text, ...], Field(max_length=32)]

    @field_validator('unresolved')
    @classmethod
    def nonblank(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError('blank unresolved concern')
        return values


def validate_stage(schema, data, context):
    # Media metadata is authoritative; tolerate decimal serialization at <=1 µs,
    # never rescale a model's event timeline or accept a materially wrong duration.
    if schema in (GroundedEvents, ReviewedRevision) and isinstance(data, dict):
        actual, reported = context['duration'], data.get('duration')
        if type(reported) in (int, float) and math.isfinite(reported) and abs(reported - actual) <= 1e-6:
            data = {**data, 'duration': actual}
            if isinstance(data.get('events'), (list, tuple)):
                events = []
                for event in data['events']:
                    if isinstance(event, dict):
                        end = event.get('end')
                        if type(end) in (int, float) and math.isfinite(end) and actual < end <= actual + 1e-6:
                            event = {**event, 'end': actual}
                    events.append(event)
                data['events'] = events
    value = schema.model_validate(data)
    if isinstance(value, GroundedEvents) and value.duration != context['duration']:
        raise ValueError('duration must match original video')
    if isinstance(value, (EvidenceReview, RevisionVerification)):
        for concern in (value.concerns if isinstance(value, EvidenceReview) else value.new_issues):
            if (concern.end > context['duration'] or
                (concern.target and concern.target.reference_time > context['duration']) or
                any(i >= context['event_count'] for i in concern.event_indices)):
                raise ValueError('review references invalid event or time')
    if isinstance(value, SamTargetPlan) and value.target:
        if not context['window_start'] <= value.target.reference_time <= context['window_end']:
            raise ValueError('SAM reference outside requested window')
    if isinstance(value, SamPromptReview) and value.correction:
        if not context['window_start'] <= value.correction.reference_time <= context['window_end']:
            raise ValueError('corrected reference outside requested window')
    if isinstance(value, (ReviewedRevision, RevisionVerification)):
        entries = value.responses if isinstance(value, ReviewedRevision) else value.verdicts
        events = value.model_dump(mode='json')['events'] if isinstance(value, ReviewedRevision) else context.get('candidate_events', [])
        issues = []
        for entry in entries:
            assessment = entry.assessment
            if (isinstance(value, RevisionVerification) and
                    entry.concern_id in context.get('assessment_required_ids', []) and assessment is None):
                issues.append(f'assessment required for {entry.concern_id}; independently classify the current candidate; use no_source_limitation if no source limitation remains')
            if assessment is not None:
                if (assessment.end > context['duration'] or assessment.event_index >= len(events)
                        or any(t > context['duration'] for t in assessment.visual_times)):
                    issues.append(f'assessment for {entry.concern_id} references invalid event or source-video time')
                    continue
                event = events[assessment.event_index]
                if assessment.candidate_quote not in event['description']:
                    issues.append(f'assessment for {entry.concern_id}: candidate_quote must be an exact contiguous phrase from your CURRENT returned event {assessment.event_index}, not an old draft or rationale')
                if assessment.start >= event['end'] or assessment.end <= event['start']:
                    issues.append(f'assessment for {entry.concern_id} [{assessment.start}, {assessment.end}] must overlap current event {assessment.event_index} [{event["start"]}, {event["end"]}]')
        ids = [entry.concern_id for entry in entries]
        if len(ids) != len(set(ids)) or set(ids) != set(context['concern_ids']):
            expected, actual = set(context['concern_ids']), set(ids)
            duplicates = sorted(i for i in actual if ids.count(i) > 1)
            issues.append('every concern must be addressed exactly once; '
                             f'missing={sorted(expected - actual)[:12]}; '
                             f'duplicate={duplicates[:12]}; unexpected={sorted(actual - expected)[:12]}')
        if isinstance(value, ReviewedRevision):
            if any(t > context['duration'] for entry in entries for t in entry.visual_times):
                issues.append('visual reference outside original video')
        if issues:
            raise CrossFieldValidationError(issues)
    return value.model_dump(mode='json')


STAGE_SCHEMAS = {'grounded_draft': GroundedEvents,
                 'evidence_review': EvidenceReview, 'sam_target_plan': SamTargetPlan,
                 'sam_prompt_review': SamPromptReview, 'grounded_revision': ReviewedRevision,
                 'revision_verification': RevisionVerification}
SCHEMAS = {schema.__name__: schema for schema in STAGE_SCHEMAS.values()}
