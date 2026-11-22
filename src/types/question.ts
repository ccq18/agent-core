export interface QuestionOption {
  label: string;
  description?: string;
}

export interface QuestionItem {
  id: string;
  question: string;
  options?: QuestionOption[];
  multiSelect?: boolean;
}

export interface PendingQuestionRequest {
  id: string;
  sessionId: string;
  runId: string;
  toolCallId: string;
  questions: QuestionItem[];
  createdAt: number;
}

export type QuestionAnswer = Record<string, string | string[]>;
