"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { CONCEPTS, type ConceptId } from "@/lib/concepts"

type Status = "idle" | "processing" | "done" | "error"

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [concept, setConcept] = useState<ConceptId | null>(null)
  const [status, setStatus] = useState<Status>("idle")
  const [outputUrl, setOutputUrl] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const audioRef = useRef<HTMLAudioElement>(null)

  // 입력 파일 미리듣기용 object URL
  const inputUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])

  useEffect(() => {
    return () => {
      if (inputUrl) URL.revokeObjectURL(inputUrl)
    }
  }, [inputUrl])

  // 생성 결과 object URL 정리
  useEffect(() => {
    return () => {
      if (outputUrl) URL.revokeObjectURL(outputUrl)
    }
  }, [outputUrl])

  // 결과가 준비되면 자동 재생
  useEffect(() => {
    if (status === "done" && outputUrl && audioRef.current) {
      audioRef.current.play().catch(() => {
        // 브라우저 정책으로 자동 재생이 막힌 경우, 사용자가 직접 재생
      })
    }
  }, [status, outputUrl])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null
    setFile(selected)
    setStatus("idle")
    setErrorMessage(null)
    if (outputUrl) {
      URL.revokeObjectURL(outputUrl)
      setOutputUrl(null)
    }
  }

  const handleGenerate = async () => {
    if (!file || !concept) return

    const backendUrl = process.env.NEXT_PUBLIC_ML_BACKEND_URL
    if (!backendUrl) {
      setErrorMessage(
        "NEXT_PUBLIC_ML_BACKEND_URL이 설정되지 않았습니다. backend/README.md를 참고해주세요."
      )
      setStatus("error")
      return
    }

    setStatus("processing")
    setErrorMessage(null)
    if (outputUrl) {
      URL.revokeObjectURL(outputUrl)
      setOutputUrl(null)
    }

    try {
      const formData = new FormData()
      formData.append("audio", file)
      formData.append("concept", concept)

      // 변환 서버로 직접 업로드한다 (Vercel 서버리스 함수의 요청 본문 크기
      // 제한을 피하기 위해 /api 라우트를 거치지 않음).
      let res: Response
      try {
        res = await fetch(`${backendUrl}/convert`, {
          method: "POST",
          body: formData,
        })
      } catch {
        throw new Error("변환 서버에 연결할 수 없습니다.")
      }

      if (!res.ok) {
        const data = await res.json().catch(() => null)
        throw new Error(
          data?.detail ?? data?.error ?? `요청에 실패했습니다 (status ${res.status})`
        )
      }

      const blob = await res.blob()
      setOutputUrl(URL.createObjectURL(blob))
      setStatus("done")
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
      )
      setStatus("error")
    }
  }

  const canGenerate = !!file && !!concept && status !== "processing"

  return (
    <main className="min-h-screen bg-[#050816] text-zinc-100 flex flex-col items-center px-6 py-16 gap-14">
      <header className="text-center space-y-2 max-w-xl">
        <h1 className="text-3xl font-semibold tracking-tight text-cyan-300">
          Sufi → Techno
        </h1>
        <p className="text-sm text-zinc-400">
          수피즘 음악을 업로드하고, 원하는 테크노 컨셉을 선택해 변환하세요.
        </p>
      </header>

      {/* 1. 파일 업로드 */}
      <section className="w-full max-w-xl space-y-3">
        <StepLabel n={1} title="수피즘 음악 업로드" />

        <label
          className="
            flex flex-col items-center justify-center
            w-full min-h-40 rounded-2xl border border-dashed border-cyan-500/30
            bg-cyan-500/5 hover:bg-cyan-500/10 transition-colors
            cursor-pointer text-center px-6 py-8
          "
        >
          <input
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={handleFileChange}
          />
          {file ? (
            <div className="space-y-1">
              <div className="text-sm font-medium text-cyan-200">{file.name}</div>
              <div className="text-xs text-zinc-500">
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <div className="text-sm text-zinc-300">
                클릭하여 오디오 파일을 선택하세요
              </div>
              <div className="text-xs text-zinc-500">MP3, WAV, FLAC 등</div>
            </div>
          )}
        </label>

        {inputUrl && (
          <audio controls src={inputUrl} className="w-full">
            <track kind="captions" />
          </audio>
        )}
      </section>

      {/* 2. 컨셉 선택 */}
      <section className="w-full max-w-3xl space-y-3">
        <StepLabel n={2} title="테크노 컨셉 선택" />

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {CONCEPTS.map((c) => {
            const selected = concept === c.id
            return (
              <motion.button
                key={c.id}
                type="button"
                onClick={() => setConcept(c.id)}
                whileTap={{ scale: 0.97 }}
                className="
                  flex flex-col items-start text-left
                  rounded-2xl border px-4 py-4
                  backdrop-blur-xl transition-shadow
                "
                style={{
                  borderColor: selected ? c.color : `${c.color}33`,
                  background: selected ? `${c.color}1f` : `${c.color}0d`,
                  boxShadow: selected ? `0 0 20px ${c.color}` : "none",
                }}
              >
                <div className="text-sm font-semibold text-white">{c.label}</div>
                <div className="text-xs text-zinc-400">{c.labelKo}</div>
                <div className="text-xs text-zinc-500 mt-2">{c.description}</div>
              </motion.button>
            )
          })}
        </div>
      </section>

      {/* 3. 생성 */}
      <section className="w-full max-w-xl flex flex-col items-center gap-3">
        <button
          type="button"
          onClick={handleGenerate}
          disabled={!canGenerate}
          className="
            px-8 py-3 rounded-full font-semibold
            bg-cyan-400 text-[#050816]
            disabled:opacity-30 disabled:cursor-not-allowed
            hover:bg-cyan-300 transition-colors
          "
        >
          {status === "processing" ? "변환 중..." : "테크노로 변환하기"}
        </button>

        {status === "error" && errorMessage && (
          <p className="text-sm text-red-400">{errorMessage}</p>
        )}
      </section>

      {/* 4. 결과 */}
      <AnimatePresence>
        {status === "done" && outputUrl && (
          <motion.section
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="w-full max-w-xl space-y-3"
          >
            <StepLabel n={4} title="생성된 트랙" />
            <audio ref={audioRef} controls autoPlay src={outputUrl} className="w-full">
              <track kind="captions" />
            </audio>
            <a
              href={outputUrl}
              download={`techno-${concept}.wav`}
              className="inline-block text-sm text-cyan-300 underline underline-offset-4 hover:text-cyan-200"
            >
              다운로드
            </a>
          </motion.section>
        )}
      </AnimatePresence>
    </main>
  )
}

function StepLabel({ n, title }: { n: number; title: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="
          flex items-center justify-center
          w-6 h-6 rounded-full text-xs font-semibold
          bg-cyan-500/10 border border-cyan-500/30 text-cyan-300
        "
      >
        {n}
      </span>
      <h2 className="text-sm font-medium text-zinc-200">{title}</h2>
    </div>
  )
}
