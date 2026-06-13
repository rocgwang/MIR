import { NextRequest, NextResponse } from "next/server"
import { isConceptId } from "@/lib/concepts"

// API 계약
// 요청: multipart/form-data
//   - audio: 변환할 수피즘 음악 파일
//   - concept: lib/concepts.ts 의 CONCEPT_IDS 중 하나
// 응답(성공): 변환된 오디오 바이너리 (Content-Type: audio/*)
// 응답(실패): { error: string } JSON, 4xx/5xx

export async function POST(request: NextRequest) {
  const formData = await request.formData()

  const audio = formData.get("audio")
  const concept = formData.get("concept")

  if (!(audio instanceof File) || audio.size === 0) {
    return NextResponse.json(
      { error: "오디오 파일이 필요합니다." },
      { status: 400 }
    )
  }

  if (!isConceptId(concept)) {
    return NextResponse.json(
      { error: "유효하지 않은 컨셉입니다." },
      { status: 400 }
    )
  }

  // ---------------------------------------------------------------------
  // TODO(backend): 여기에 실제 음악 생성 모델 연동 코드를 작성합니다.
  //   입력: audio (File), concept (ConceptId)
  //   출력: 변환된 오디오의 바이너리(ArrayBuffer/Buffer)와 MIME 타입
  // 아래의 echo 구현은 모델이 도착하기 전까지 프런트엔드 플로우
  // (업로드 -> 컨셉 선택 -> 생성 -> 재생)를 테스트하기 위한 임시 동작입니다.
  // ---------------------------------------------------------------------
  const outputBuffer = await audio.arrayBuffer()
  const outputType = audio.type || "audio/mpeg"

  return new NextResponse(outputBuffer, {
    status: 200,
    headers: {
      "Content-Type": outputType,
      "Content-Disposition": `inline; filename="techno-${concept}-${audio.name}"`,
    },
  })
}
