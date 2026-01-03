import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  SafeAreaView,
  View,
  Text,
  TextInput,
  Pressable,
  ActivityIndicator,
  ScrollView,
  StyleSheet,
} from "react-native";
import Slider from "@react-native-community/slider";
import BottomSheet, { BottomSheetView } from "@gorhom/bottom-sheet";

const API_BASE = "https://musicophile-ai.onrender.com";

type SongHit = {
  id: number;
  song: string;
  artist: string;
  genre?: string | null;
  emotion?: string | null;
  search_score?: number;
};

type RecItem = {
  id: number;
  song: string;
  artist: string;
  genre?: string | null;
  emotion?: string | null;
  score: number;
  audio_sim: number;
  lyric_sim: number;
  meta_sim: number;
};

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function HighlightedText({
  text,
  query,
  style,
}: {
  text: string;
  query: string;
  style?: any;
}) {
  const q = query.trim();
  if (!q) return <Text style={style}>{text}</Text>;

  const re = new RegExp(`(${escapeRegExp(q)})`, "gi");
  const parts = text.split(re);

  return (
    <Text style={style}>
      {parts.map((part, i) => {
        const isMatch = part.toLowerCase() === q.toLowerCase();
        return isMatch ? (
          <Text key={i} style={[style, styles.bold]}>
            {part}
          </Text>
        ) : (
          <Text key={i} style={style}>
            {part}
          </Text>
        );
      })}
    </Text>
  );
}

function clamp(s: string, max = 60) {
  if (!s) return "";
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

function formatPct(x: number) {
  return `${Math.round(x * 100)}%`;
}

export default function HomeScreen() {
  const [showResults, setShowResults] = useState(true);

  // split search
  const [titleQuery, setTitleQuery] = useState("");
  const [artistQuery, setArtistQuery] = useState("");

  const [results, setResults] = useState<SongHit[]>([]);
  const [loadingSearch, setLoadingSearch] = useState(false);

  const [seed, setSeed] = useState<SongHit | null>(null);

  // rec controls
  const [wAudio, setWAudio] = useState(0.4);
  const [wLyrics, setWLyrics] = useState(0.4);
  const [wMeta, setWMeta] = useState(0.2);
  const [diversify, setDiversify] = useState(0.0);
  const [topK, setTopK] = useState(10);

  // ONLY two options now
  const [artistMode, setArtistMode] = useState<"same" | "different">("different");

  // rec results
  const [loadingRec, setLoadingRec] = useState(false);
  const [recs, setRecs] = useState<RecItem[]>([]);
  const [recError, setRecError] = useState<string | null>(null);

  // Bottom sheet
  const sheetRef = useRef<BottomSheet>(null);
  const snapPoints = useMemo(() => ["30%", "75%"], []);
  const openSheet = useCallback(() => sheetRef.current?.snapToIndex(1), []);
  const closeSheet = useCallback(() => sheetRef.current?.close(), []);

  const normWeights = useMemo(() => {
    const s = wAudio + wLyrics + wMeta;
    if (s <= 0) return { a: 0.4, l: 0.4, m: 0.2 };
    return { a: wAudio / s, l: wLyrics / s, m: wMeta / s };
  }, [wAudio, wLyrics, wMeta]);

  // search
  useEffect(() => {
    const t = setTimeout(async () => {
      const qt = titleQuery.trim();
      const qa = artistQuery.trim();

      if (!qt && !qa) {
        setResults([]);
        return;
      }

      setLoadingSearch(true);
      try {
        const url =
          `${API_BASE}/search?` +
          `q_title=${encodeURIComponent(qt)}&` +
          `q_artist=${encodeURIComponent(qa)}&` +
          `n=30`;

        const res = await fetch(url);
        const data = await res.json();
        setResults(data.results || []);
      } catch (e) {
        console.log("Search error:", e);
      } finally {
        setLoadingSearch(false);
      }
    }, 250);

    return () => clearTimeout(t);
  }, [titleQuery, artistQuery]);

  async function runRecommend() {
    if (!seed) return;
    setLoadingRec(true);
    setRecError(null);

    try {
      const body = {
        seed_id: seed.id,
        w_audio: normWeights.a,
        w_lyrics: normWeights.l,
        w_meta: normWeights.m,
        diversify_genre: diversify,
        top_k: topK,
        artist_mode: artistMode, // only same/different
      };

      const res = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);

      setRecs(data.recommendations || []);
      closeSheet();
    } catch (e: any) {
      console.log("Recommend error:", e);
      setRecError(e?.message || "Failed to recommend");
    } finally {
      setLoadingRec(false);
    }
  }

  function artistModeLabel(m: "same" | "different") {
    return m === "same" ? "Same artist only" : "Exclude seed artist";
  }

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.container}>
        <Text style={styles.title}>musicophile.ai</Text>
        <Text style={styles.subtitle}>Search a seed song. Add artist to make it exact.</Text>

        {/* Pick seed */}
        <View style={styles.card}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>Pick a seed</Text>
            <Text style={styles.rowMeta}>Title + (optional) artist</Text>
          </View>

          <View style={{ padding: 12 }}>
            <TextInput
              value={titleQuery}
              onChangeText={setTitleQuery}
              placeholder="Song title (e.g., Blinding Lights)"
              placeholderTextColor="#6b7280"
              style={styles.input}
              autoCorrect={false}
              autoCapitalize="none"
            />

            <TextInput
              value={artistQuery}
              onChangeText={setArtistQuery}
              placeholder="Artist (optional, e.g., Drake)"
              placeholderTextColor="#6b7280"
              style={[styles.input, { marginTop: 10 }]}
              autoCorrect={false}
              autoCapitalize="none"
            />

            {loadingSearch && <ActivityIndicator style={{ marginTop: 12 }} />}

            {!loadingSearch && (titleQuery.trim() || artistQuery.trim()) && results.length === 0 && (
              <Text style={[styles.rowMeta, { marginTop: 10 }]}>No matches.</Text>
            )}

            {results.length > 0 && (
              <Pressable onPress={() => setShowResults((v) => !v)} style={[styles.toggleRow, { marginTop: 12 }]}>
                <Text style={{ color: "white", fontWeight: "800" }}>
                  {showResults ? "Hide results" : "Show results"}
                </Text>
                <Text style={{ color: "#9ca3af", fontSize: 18 }}>{showResults ? "▲" : "▼"}</Text>
              </Pressable>
            )}

            {showResults && results.length > 0 && (
              <View style={{ marginTop: 10 }}>
                {results.slice(0, 12).map((item) => (
                  <Pressable
                    key={item.id}
                    onPress={() => {
                      setSeed(item);
                      setRecs([]);
                      setShowResults(false);
                      openSheet();
                    }}
                    style={styles.row}
                  >
                    <HighlightedText text={item.song} query={titleQuery} style={styles.rowTitle} />
                    <Text style={styles.rowSub}>
                      by{" "}
                      <HighlightedText text={clamp(item.artist, 80)} query={artistQuery} style={styles.rowSub} />
                    </Text>
                    <Text style={styles.rowMeta}>
                      {item.genre ?? "Unknown"} • {item.emotion ?? "Unknown"}
                    </Text>
                  </Pressable>
                ))}
              </View>
            )}
          </View>
        </View>

        {/* Selected seed */}
        {seed ? (
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>Selected seed</Text>
              <Pressable onPress={openSheet}>
                <Text style={styles.linkText}>Tune ▸</Text>
              </Pressable>
            </View>
            <View style={{ padding: 12 }}>
              <Text style={styles.rowTitle}>{seed.song}</Text>
              <Text style={styles.rowSub}>by {clamp(seed.artist, 80)}</Text>
              <Text style={styles.rowMeta}>
                {seed.genre ?? "Unknown genre"} • {seed.emotion ?? "Unknown emotion"}
              </Text>
              <Text style={[styles.rowMeta, { marginTop: 4 }]}>
                Artist mode: {artistModeLabel(artistMode)}
              </Text>
            </View>
          </View>
        ) : (
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.cardTitle}>Pick a seed song</Text>
              <Text style={styles.rowMeta}>Then you can tune recommendations</Text>
            </View>
          </View>
        )}

        {/* Recommendations */}
        {recs.length > 0 && (
          <View style={{ marginTop: 16 }}>
            <Text style={styles.sectionTitle}>Recommendations</Text>
            {recs.map((r, idx) => (
              <View key={r.id} style={styles.recCard}>
                <Text style={styles.recTitle}>
                  {idx + 1}. {r.song}
                </Text>
                <Text style={styles.rowSub}>by {clamp(r.artist, 80)}</Text>
                <Text style={styles.rowMeta}>
                  {r.genre ?? "Unknown"} • {r.emotion ?? "Unknown"}
                </Text>
                <Text style={styles.recMeta}>
                  score {r.score.toFixed(3)} • audio {r.audio_sim.toFixed(3)} • lyrics{" "}
                  {r.lyric_sim.toFixed(3)} • meta {r.meta_sim.toFixed(3)}
                </Text>
              </View>
            ))}
          </View>
        )}

        <View style={{ height: 140 }} />
      </ScrollView>

      {/* Sticky bar */}
      <View style={styles.stickyBar}>
        <Pressable
          disabled={!seed || loadingRec}
          onPress={seed ? openSheet : undefined}
          style={[styles.secondaryBtn, (!seed || loadingRec) && { opacity: 0.6 }]}
        >
          <Text style={styles.secondaryBtnText}>{seed ? "Tune sliders" : "Select a seed first"}</Text>
        </Pressable>

        <Pressable
          disabled={!seed || loadingRec}
          onPress={runRecommend}
          style={[styles.primaryBtn, (!seed || loadingRec) && styles.primaryBtnDisabled]}
        >
          <Text style={styles.primaryBtnText}>{loadingRec ? "Finding..." : "Find similar songs"}</Text>
        </Pressable>

        {recError && <Text style={styles.err}>{recError}</Text>}
      </View>

      {/* Bottom Sheet */}
      <BottomSheet
        ref={sheetRef}
        index={-1}
        snapPoints={snapPoints}
        enablePanDownToClose
        backgroundStyle={{ backgroundColor: "#0f172a" }}
        handleIndicatorStyle={{ backgroundColor: "#475569" }}
      >
        <BottomSheetView style={{ paddingHorizontal: 16, paddingBottom: 18 }}>
          <View style={styles.sheetHeader}>
            <Text style={styles.sheetTitle}>Tune recommendations</Text>
            <Pressable onPress={closeSheet}>
              <Text style={styles.linkText}>Close</Text>
            </Pressable>
          </View>

          <Text style={styles.sheetSub}>
            Auto-normalized: Audio {formatPct(normWeights.a)} • Lyrics {formatPct(normWeights.l)} • Meta{" "}
            {formatPct(normWeights.m)}
          </Text>

          <Text style={[styles.sliderLabel, { marginTop: 18 }]}>Artist</Text>
          <View style={styles.segment}>
            <Pressable
              onPress={() => setArtistMode("different")}
              style={[styles.segmentBtn, artistMode === "different" && styles.segmentBtnActive]}
            >
              <Text style={[styles.segmentText, artistMode === "different" && styles.segmentTextActive]}>
                Exclude
              </Text>
            </Pressable>

            <Pressable
              onPress={() => setArtistMode("same")}
              style={[styles.segmentBtn, artistMode === "same" && styles.segmentBtnActive]}
            >
              <Text style={[styles.segmentText, artistMode === "same" && styles.segmentTextActive]}>
                Same
              </Text>
            </Pressable>
          </View>

          <Text style={styles.sliderLabel}>🎧 Audio: {formatPct(normWeights.a)}</Text>
          <Slider minimumValue={0} maximumValue={1} value={wAudio} onValueChange={setWAudio} step={0.01}
            minimumTrackTintColor="#60a5fa" maximumTrackTintColor="#374151" />

          <Text style={styles.sliderLabel}>✍️ Lyrics: {formatPct(normWeights.l)}</Text>
          <Slider minimumValue={0} maximumValue={1} value={wLyrics} onValueChange={setWLyrics} step={0.01}
            minimumTrackTintColor="#34d399" maximumTrackTintColor="#374151" />

          <Text style={styles.sliderLabel}>🧠 Meta: {formatPct(normWeights.m)}</Text>
          <Slider minimumValue={0} maximumValue={1} value={wMeta} onValueChange={setWMeta} step={0.01}
            minimumTrackTintColor="#fbbf24" maximumTrackTintColor="#374151" />

          <Text style={styles.sliderLabel}>🌈 Diversify genre: {formatPct(diversify)}</Text>
          <Text style={styles.helperText}>Higher = penalize same-genre songs more</Text>
          <Slider minimumValue={0} maximumValue={1} value={diversify} onValueChange={setDiversify} step={0.01}
            minimumTrackTintColor="#fb7185" maximumTrackTintColor="#374151" />

          <Text style={styles.sliderLabel}>Top results: {topK}</Text>
          <Slider minimumValue={5} maximumValue={25} value={topK} onValueChange={(v) => setTopK(Math.round(v))}
            step={1} minimumTrackTintColor="#a78bfa" maximumTrackTintColor="#374151" />

          <Pressable
            disabled={!seed || loadingRec}
            onPress={runRecommend}
            style={[styles.primaryBtn, { marginTop: 14 }, (!seed || loadingRec) && styles.primaryBtnDisabled]}
          >
            <Text style={styles.primaryBtnText}>{loadingRec ? "Finding..." : "Find similar songs"}</Text>
          </Pressable>
        </BottomSheetView>
      </BottomSheet>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: "#0b0b0b" },
  container: { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 160, maxWidth: 520, alignSelf: "center", width: "100%" },
  title: { fontSize: 28, fontWeight: "900", color: "white" },
  subtitle: { marginTop: 6, color: "#a1a1aa" },

  input: { paddingVertical: 12, paddingHorizontal: 14, borderWidth: 1, borderColor: "#1f2937", borderRadius: 14, color: "white", backgroundColor: "#0f0f0f", fontSize: 16 },

  card: { marginTop: 12, backgroundColor: "#111827", borderWidth: 1, borderColor: "#1f2937", borderRadius: 16, overflow: "hidden" },
  cardHeader: { padding: 12, borderBottomWidth: 1, borderBottomColor: "#1f2937", flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  cardTitle: { color: "white", fontWeight: "900", fontSize: 14 },
  linkText: { color: "#60a5fa", fontWeight: "800" },

  toggleRow: { paddingVertical: 10, paddingHorizontal: 12, borderRadius: 12, backgroundColor: "#0f0f0f", borderWidth: 1, borderColor: "#1f2937", flexDirection: "row", justifyContent: "space-between", alignItems: "center" },

  row: { paddingVertical: 12, paddingHorizontal: 12, borderBottomWidth: 1, borderBottomColor: "#1f2937" },
  rowTitle: { color: "white", fontWeight: "800", fontSize: 15 },
  rowSub: { color: "#cbd5e1", marginTop: 2 },
  rowMeta: { color: "#94a3b8", marginTop: 6, fontSize: 12 },

  bold: { fontWeight: "900" },

  sectionTitle: { color: "white", fontWeight: "900", fontSize: 16, marginTop: 12 },

  recCard: { marginTop: 10, padding: 12, borderRadius: 16, backgroundColor: "#0b1220", borderWidth: 1, borderColor: "#1f2937" },
  recTitle: { color: "white", fontWeight: "900" },
  recMeta: { color: "#9ca3af", marginTop: 8 },

  stickyBar: { position: "absolute", left: 0, right: 0, bottom: 0, padding: 12, backgroundColor: "rgba(11,11,11,0.92)", borderTopWidth: 1, borderTopColor: "#1f2937", gap: 10 },

  primaryBtn: { backgroundColor: "#2563eb", borderRadius: 14, paddingVertical: 14, alignItems: "center" },
  primaryBtnDisabled: { backgroundColor: "#1f2937" },
  primaryBtnText: { color: "white", fontWeight: "900", fontSize: 16 },

  secondaryBtn: { borderRadius: 14, paddingVertical: 12, alignItems: "center", borderWidth: 1, borderColor: "#1f2937", backgroundColor: "#0f0f0f" },
  secondaryBtnText: { color: "#e5e7eb", fontWeight: "800" },

  sheetHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  sheetTitle: { color: "white", fontWeight: "900", fontSize: 16 },
  sheetSub: { color: "#9ca3af" },

  sliderLabel: { color: "white", marginTop: 14, fontWeight: "800" },
  helperText: { color: "#94a3b8", marginTop: 2, marginBottom: 6 },

  segment: { flexDirection: "row", backgroundColor: "#0b0b0b", borderWidth: 1, borderColor: "#1f2937", borderRadius: 14, overflow: "hidden", marginTop: 8 },
  segmentBtn: { flex: 1, paddingVertical: 10, alignItems: "center" },
  segmentBtnActive: { backgroundColor: "#1e293b" },
  segmentText: { color: "#cbd5e1", fontWeight: "800", fontSize: 12 },
  segmentTextActive: { color: "white" },

  err: { color: "#fb7185", marginTop: 8 },
});
