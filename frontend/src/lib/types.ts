/** The structured part of a Coordinator reply (mirrors coordinator_agent/reply.py). */
export type ReplyKind =
  | "emergency"
  | "question"
  | "report"
  | "slots"
  | "booked"
  | "answer"
  | "no_answer"
  | "greeting"
  | "out_of_scope";

export interface Source {
  number: number;
  question: string;
  answer_id: string;
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
  urgency: string;
  status: string;
}

export interface Report {
  title: string;
  notice: string;
  summary: string;
  case: Record<string, string>;
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
  report: Report | null;
  offer: Offer | null;
  slots: Slot[];
  appointment: Appointment | null;
}

/** A button press, sent to the Coordinator alongside (or instead of) text. */
export type ChatAction =
  | { type: "show_slots"; clinician_type?: string }
  | { type: "book"; start: string; clinician_type?: string };

/** What /api/chat streams to the browser, one JSON object per event. */
export type ChatEvent =
  | { type: "progress"; text: string }
  | { type: "reply"; text: string; data: ReplyData | null; contextId: string; state: string }
  | { type: "error"; text: string };

/** One booked visit as the clinician page shows it. */
export interface ClinicAppointment extends Appointment {
  booking_uid: string;
  report: Report;
  created_at: string;
}
