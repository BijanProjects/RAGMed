/** The structured part of a Coordinator reply (mirrors coordinator_agent/reply.py). */
export type ReplyKind =
  | "emergency"
  | "question"
  | "report"
  | "slots"
  | "verify_email"
  | "booked"
  | "answer"
  | "no_answer"
  | "greeting"
  | "out_of_scope";

export interface Source {
  number: number;
  question: string;
  answer_id: string;
  excerpt?: string; // the passage text the answer relied on
  url?: string; // the NIH page it comes from, when it still loads
}

export interface Offer {
  clinician_type: string;
  visit_minutes: number;
  urgency: string;
  alternative: string | null;
}

export interface Slot {
  start: string;
  end: string;
  physician: string;
  clinician_type: string;
  visit_minutes: number;
}

export interface Appointment {
  reference: string;
  start: string;
  end: string;
  minutes: number;
  clinician_type: string;
  physician: string;
  patient_name?: string;
  urgency: string;
  status: string;
}

/** A booking waiting for the code emailed to the patient. */
export interface Verification {
  email: string;
  start: string;
  clinician_type: string;
  minutes: number; // how long the code is valid
}

export interface Report {
  title: string;
  notice: string;
  summary: string;
  case: Record<string, string>;
  /** The Care Coordinator's handoff to the clinician (missing on older reports). */
  impression?: string;
  open_questions?: string[];
  urgency_score?: number; // 1 (emergency) to 5 (self-care)
  score_reason?: string;
  assessment: string;
  possible_conditions: string[];
  urgency: string;
  clinician_type: string;
  visit_minutes: number;
  visit_reason: string;
  sources: Source[];
  sufficient_evidence: boolean;
  created_at: string;
}

export interface ReplyData {
  kind: ReplyKind;
  text: string;
  urgency: string | null;
  recommend_visit: boolean;
  possible_conditions: string[];
  sources: Source[];
  red_flag: string | null;
  quick_replies: string[];
  multi_select?: boolean; // the patient may tick several quick replies
  report: Report | null;
  offer: Offer | null;
  slots: Slot[];
  appointment: Appointment | null;
  verification?: Verification | null;
}

/** A button press, sent to the Coordinator alongside (or instead of) text. */
export type ChatAction =
  | { type: "show_slots"; clinician_type?: string }
  | { type: "book"; start: string; clinician_type?: string; name: string; email: string } // emails a code
  | { type: "resend" }
  | { type: "confirm"; code: string };

/** One step of the care team's work: which agent did what. */
export interface Step {
  agent: string; // "Care Coordinator", "Diagnosis agent", "Knowledge base", "Scheduling"
  text: string;
}

/** What /api/chat streams to the browser, one JSON object per event. */
export type ChatEvent =
  | { type: "progress"; text: string; agent?: string }
  /** A piece of the reply while it is written; `reset` starts it again. */
  | { type: "draft"; text: string; label?: string; reset?: boolean }
  | { type: "reply"; text: string; data: ReplyData | null; contextId: string; state: string }
  | { type: "error"; text: string };

/** One booked visit as the clinician page shows it. */
export interface ClinicAppointment extends Appointment {
  booking_uid: string;
  report: Report;
  created_at: string;
}
