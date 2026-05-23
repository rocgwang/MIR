"use client"

import { motion } from "framer-motion"
import { useRef, useState } from "react"

const tracks = [
  { title: "Desert Wind", color: "#FFB347" },
  { title: "Summer Vibes", color: "#FF69B4" },
  { title: "Deep Ocean", color: "#4FD1FF" },
  { title: "Electric Dream", color: "#FFD84D" },
  { title: "Midnight Jazz", color: "#A0AEC0" },
  { title: "Urban Beat", color: "#C08457" },
  { title: "Forest Whisper", color: "#7DD3FC" },
  { title: "Stellar Flow", color: "#C084FC" },
]

export default function TrackWheel({
  label,
}: {
  label: string
}) {
  const [rotation, setRotation] = useState(0)

  const dragging = useRef(false)
  const lastX = useRef(0)

  const handleMouseDown = (
    e: React.MouseEvent<HTMLDivElement>
  ) => {
    dragging.current = true
    lastX.current = e.clientX
  }

  const handleMouseMove = (
    e: React.MouseEvent<HTMLDivElement>
  ) => {
    if (!dragging.current) return

    const delta = e.clientX - lastX.current

    setRotation((prev) => prev + delta * 0.4)

    lastX.current = e.clientX
  }

  const handleMouseUp = () => {
    dragging.current = false
  }

  return (
    <div
      className="
        relative
        flex
        items-center
        justify-center
        w-full
        h-screen
        overflow-hidden
        bg-[#050816]
      "
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <div
        className="
          relative
          w-[700px]
          h-[700px]
          rounded-full
          border
          border-cyan-500/20
        "
        onMouseDown={handleMouseDown}
      >
        {/* 중앙 원 */}
        <div
          className="
            absolute
            left-1/2
            top-1/2
            -translate-x-1/2
            -translate-y-1/2
            w-52
            h-52
            rounded-full
            border
            border-cyan-500/20
            bg-cyan-500/5
            flex
            items-center
            justify-center
            text-center
            text-zinc-400
            text-lg
            backdrop-blur-md
          "
        >
        {label}
        <br />
        선택
        </div>

        {/* 선택 포인터 */}
        <div
          className="
            absolute
            left-1/2
            top-0
            -translate-x-1/2
            flex
            flex-col
            items-center
          "
        >
          <div className="w-1 h-12 bg-cyan-300" />

          <div
            className="
              w-4
              h-4
              rounded-full
              bg-cyan-300
            "
            style={{
              boxShadow: "0 0 20px #67e8f9",
            }}
          />
        </div>

        {/* 카드 */}
        {tracks.map((track, index) => {
          const angle =
            ((360 / tracks.length) * index + rotation) *
            (Math.PI / 180)

          const radius = 260

          const x = radius * Math.cos(angle)
          const y = radius * Math.sin(angle)

          return (
            <motion.div
              key={track.title}
              className="
                absolute
                w-32
                h-20
                rounded-2xl
                backdrop-blur-xl
                border
                cursor-pointer
                select-none
                flex
                flex-col
                justify-center
                px-4
              "
              animate={{
                scale: 1,
              }}
              style={{
                left: `calc(50% + ${x}px - 64px)`,
                top: `calc(50% + ${y}px - 40px)`,
                borderColor: track.color,
                background: `${track.color}15`,
                boxShadow: `0 0 20px ${track.color}`,
              }}
            >
              <div className="text-sm font-semibold text-white">
                {track.title}
              </div>

              <div className="text-xs text-zinc-400 mt-1">
                Track
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}