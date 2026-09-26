# pyrefly: ignore [missing-import]
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        qs = parse_qs(parsed.query)
        video_id = qs.get("v", [None])[0]
        if video_id is None:
            raise ValueError(f"Cannot extract video ID from URL: {url}")
        return video_id
    raise ValueError(f"Cannot extract video ID from URL: {url}")


def fetch_transcript(video_url: str, language: str = "en") -> str:
    """Fetch and join the transcript for a YouTube video.

    Args:
        video_url: Full YouTube video URL.
        language: Preferred transcript language code (default: 'en').

    Returns:
        Transcript text as a single string.
    """
    video_id = extract_video_id(video_url)
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=[language])
    return " ".join(snippet.text for snippet in fetched.snippets)
