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
      <TrackWheel label="TRACK A" />

      <TrackWheel label="TRACK B" />
    </main>
  )
}