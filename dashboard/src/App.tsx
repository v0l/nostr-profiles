import { useState, useEffect, useCallback } from "preact/hooks";
import { tryParseNostrLink } from "@snort/system";
import { ProfileCard } from "./components/ProfileCard";
import {
	StatsBar,
	LabelStatsPanel,
	RecentItemCard,
} from "./components/StatsAndRecent";
import type { Profile, RecentItem as RecentItemType, Stats } from "./types";

export function App() {
	const [pubkey, setPubkey] = useState("");
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [profile, setProfile] = useState<Profile | null>(null);
	const [recent, setRecent] = useState<RecentItemType[]>([]);
	const [stats, setStats] = useState<Stats | null>(null);

	const searchProfile = useCallback(
		async (pk?: string) => {
			const key = (pk || pubkey).trim();
			if (!key) return;
			setLoading(true);
			setError(null);
			setProfile(null);
			try {
				let hexKey: string | null = null;
				const parsed = tryParseNostrLink(key);
				if (parsed) {
					if (parsed.type === "npub" || parsed.type === "nprofile") {
						hexKey = parsed.id;
					}
				}
				if (!hexKey && /^[0-9a-f]{64}$/i.test(key)) {
					hexKey = key;
				}

				if (hexKey) {
					const res = await fetch("/api/profile/" + encodeURIComponent(hexKey));
					if (res.status === 404) {
						setError("Profile not found");
						return;
					}
					if (!res.ok) {
						setError((await res.text()) || "Failed to fetch");
						return;
					}
					setProfile(await res.json());
				} else if (pk) {
					// pk is a label name passed from LabelStatsPanel
					const res = await fetch(
						"/api/search/label?label=" + encodeURIComponent(key),
					);
					if (!res.ok) {
						setError((await res.text()) || "Label search failed");
						return;
					}
					const results = await res.json();
					setRecent(results);
					if (results.length === 0) {
						setError("No profiles found with label: " + key);
						return;
					}
					const first = results[0];
					const profileRes = await fetch(
						"/api/profile/" + encodeURIComponent(first.pubkey),
					);
					if (profileRes.ok) {
						setProfile(await profileRes.json());
					}
					setRecent(results);
				} else {
					const res = await fetch("/api/search?q=" + encodeURIComponent(key));
					if (!res.ok) {
						setError((await res.text()) || "Search failed");
						return;
					}
					const results = await res.json();
					setRecent(results);
					if (results.length === 0) {
						setError("No results found");
						return;
					}
					const first = results[0];
					const profileRes = await fetch(
						"/api/profile/" + encodeURIComponent(first.pubkey),
					);
					if (profileRes.ok) {
						setProfile(await profileRes.json());
					}
					setRecent(results);
				}
				if (pk) setPubkey(pk);
			} catch {
				setError("Failed to connect to server");
			} finally {
				setLoading(false);
			}
		},
		[pubkey],
	);

	useEffect(() => {
		fetch("/api/recent?limit=20")
			.then((r) => r.json())
			.then(setRecent)
			.catch(() => {});
		const loadStats = () =>
			fetch("/api/stats")
				.then((r) => r.json())
				.then(setStats)
				.catch(() => {});
		loadStats();
		const iv = setInterval(loadStats, 10000);
		return () => clearInterval(iv);
	}, []);

	useEffect(() => {
		if (profile) {
			setRecent((prev) => {
				const idx = prev.findIndex((r) => r.pubkey === profile.pubkey);
				if (idx === -1) return prev;
				const updated = [...prev];
				updated[idx] = {
					...updated[idx],
					name: profile.name,
					display_name: profile.display_name,
					picture: profile.picture,
				};
				return updated;
			});
		}
	}, [profile]);

	return (
		<div class="app">
			<h1>Nostr Profile Classifier</h1>

			<StatsBar stats={stats} />
			<LabelStatsPanel stats={stats} onSearch={searchProfile} />

			<div class="toolbar">
				<a href="/api/download-db" class="download-link" download>
					⬇ Download Database
				</a>
			</div>

			<div class="search-box">
				<input
					type="text"
					value={pubkey}
					onInput={(e) => setPubkey((e.target as HTMLInputElement).value)}
					onKeyPress={(e) => e.key === "Enter" && searchProfile()}
					placeholder="Enter npub, nprofile, hex pubkey, or search labels..."
					disabled={loading}
				/>
				<button onClick={() => searchProfile()} disabled={loading}>
					{loading ? "Loading..." : "Search"}
				</button>
			</div>

			{error && <div class="error">{error}</div>}
			{profile && <ProfileCard profile={profile} onSearch={searchProfile} />}

			<h2>Recent Classifications</h2>
			<div>
				{recent.length === 0 && (
					<div class="loading">No classifications yet</div>
				)}
				{recent.map((item) => (
					<RecentItemCard
						key={item.pubkey}
						item={item}
						onSearch={searchProfile}
					/>
				))}
			</div>
		</div>
	);
}
