/** The structured part of a Coordinator reply (mirrors coordinator_agent/reply.py). */
export type ReplyKind =
  | "emergency"
  | "answer"
  | "no_answer"
  | "greeting"
  | "booking"
  | "out_of_scope";

export interface Source {
  number: number;
  question: string;
  answer_id: string;
}

export interface ReplyData {
  kind: ReplyKind;
  text: string;
  urgency: string | null;
  recommend_visit: boolean;
  specialty: string | null;
  possible_conditions: string[];
  sources: Source[];
  red_flag: string | null;
}

/** What /api/chat streams to the browser, one JSON object per event. */
export type ChatEvent =
  | { type: "progress"; text: string }
  | { type: "reply"; text: string; data: ReplyData | null; contextId: string; state: string }
  | { type: "error"; text: string };
