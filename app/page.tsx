import Link from "next/link"
import TrackWheel from "@/components/TrackWheel"

export default function Home() {
  return (
    <main
      className="
        min-h-screen
        bg-[#050816]
        flex
        items-center
        justify-center
        gap-16
      "
    >
      <Link
        href="/convert"
        className="
          fixed top-6 right-6 z-10
          text-xs text-cyan-400/70 hover:text-cyan-300
          transition-colors
        "
      >
        Sufi → Techno 변환 →
      </Link>

      <TrackWheel label="TRACK A" />

      <TrackWheel label="TRACK B" />
    </main>
  )
}