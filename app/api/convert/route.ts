import { NextRequest, NextResponse } from "next/server"
import { isConceptId } from "@/lib/concepts"

// API 계약
// 요청: multipart/form-data
//   - audio: 변환할 수피즘 음악 파일
//   - concept: lib/concepts.ts 의 CONCEPT_IDS 중 하나
// 응답(성공): 변환된 오디오 바이너리 (Content-Type: audio/*)
// 응답(실패): { error: string } JSON, 4xx/5xx
//
// 실제 변환(Demucs 분리 -> MusicGen 생성 -> 믹스)은 GPU가 필요해 별도의
// Python 백엔드(backend/)에서 수행하고, 이 라우트는 ML_BACKEND_URL로
// 요청을 그대로 전달한다. 설정 방법은 backend/README.md 참고.

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

  const backendUrl = process.env.ML_BACKEND_URL
  if (!backendUrl) {
    return NextResponse.json(
      { error: "ML_BACKEND_URL이 설정되지 않았습니다. backend/README.md를 참고해 변환 서버를 실행하고 환경변수를 설정해주세요." },
      { status: 503 }
    )
  }

  const backendForm = new FormData()
  backendForm.append("audio", audio, audio.name)
  backendForm.append("concept", concept)

  let backendResponse: Response
  try {
    backendResponse = await fetch(`${backendUrl}/convert`, {
      method: "POST",
      body: backendForm,
    })
  } catch {
    return NextResponse.json(
      { error: "변환 서버에 연결할 수 없습니다." },
      { status: 502 }
    )
  }

  if (!backendResponse.ok) {
    const message = await backendResponse.text().catch(() => "")
    return NextResponse.json(
      { error: message || "변환 중 오류가 발생했습니다." },
      { status: backendResponse.status }
    )
  }

  const outputBuffer = await backendResponse.arrayBuffer()
  const outputType = backendResponse.headers.get("Content-Type") ?? "audio/wav"

  return new NextResponse(outputBuffer, {
    status: 200,
    headers: {
      "Content-Type": outputType,
      "Content-Disposition": `inline; filename="techno-${concept}.wav"`,
    },
  })
}
