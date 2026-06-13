export const CONCEPTS = [
  {
    id: "powerful",
    label: "Powerful",
    labelKo: "파워풀",
    description: "강렬한 킥과 드라이빙 베이스로 압도하는 사운드",
    color: "#FF3B6E",
  },
  {
    id: "groove",
    label: "Groove",
    labelKo: "그루브",
    description: "리듬감 있는 펑키 그루브 중심의 사운드",
    color: "#FFB347",
  },
  {
    id: "hypnotic",
    label: "Hypnotic",
    labelKo: "히프노틱",
    description: "반복되는 패턴으로 몰입을 유도하는 사운드",
    color: "#C084FC",
  },
  {
    id: "dark",
    label: "Dark Industrial",
    labelKo: "다크 인더스트리얼",
    description: "거칠고 어두운 인더스트리얼 텍스처",
    color: "#A0AEC0",
  },
  {
    id: "melodic",
    label: "Melodic",
    labelKo: "멜로딕",
    description: "신비로운 멜로디 레이어가 살아있는 사운드",
    color: "#4FD1FF",
  },
  {
    id: "minimal",
    label: "Minimal Deep",
    labelKo: "미니멀 딥",
    description: "여백을 살린 미니멀하고 깊은 그루브",
    color: "#7DD3FC",
  },
] as const

export type Concept = (typeof CONCEPTS)[number]
export type ConceptId = Concept["id"]

export const CONCEPT_IDS = CONCEPTS.map((c) => c.id) as ConceptId[]

export function isConceptId(value: unknown): value is ConceptId {
  return typeof value === "string" && (CONCEPT_IDS as string[]).includes(value)
}
