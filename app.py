import os
import json
import time
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def safe_search_extract(resp_json):
    """Return (raw_text, annotations) from a Responses API result, or raise ValueError with the API error message."""
    if "output" not in resp_json:
        err = resp_json.get("error", {})
        raise ValueError(err.get("message", str(resp_json)))
    for item in resp_json["output"]:
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text", ""), c.get("annotations", [])
    raise ValueError("No search results returned")


def web_search_call(headers, prompt, search_context_size="medium", timeout=180):
    """POST to the Responses API with the web_search tool. Returns the parsed JSON response."""
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json={
            "model": "gpt-5.6-terra",
            "tools": [{"type": "web_search", "search_context_size": search_context_size}],
            "input": prompt,
        },
        timeout=timeout,
    )
    return resp.json()


def gpt_format(headers, prompt, max_tokens=4096):
    """Call gpt-4o-mini with json_object mode and return the inner array as a JSON string."""
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return only valid JSON as {\"result\": [...]}. No markdown, no extra text."},
                {"role": "user",   "content": prompt},
            ],
        },
        timeout=120,
    )
    data = resp.json()
    if "choices" in data:
        raw = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw)
            # extract the array from whichever key wraps it
            for val in parsed.values():
                if isinstance(val, list):
                    data["choices"][0]["message"]["content"] = json.dumps(val)
                    break
        except Exception:
            pass  # leave as-is; frontend parseJSON will handle fences
    return resp.status_code, data

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
BASE = os.path.abspath(os.path.dirname(__file__))

def resolve_openai_key(data):
    """Prefer a per-request key sent from the frontend; fall back to the server's .env key, if any."""
    return (data.get("openai_key") or "").strip() or OPENAI_API_KEY

# Simple in-memory cache for event searches (avoids re-calling $0.03 search for same query)
_event_cache = {}   # key: "category|keywords" → {"ts": epoch, "data": response_json}


@app.route("/")
def serve_app():
    return send_file(os.path.join(BASE, "index.html"))


@app.route("/api/search-events", methods=["POST"])
def search_events():
    """Two-step: gpt-5.6-terra web search finds real events, gpt-4o-mini formats to JSON."""
    data = request.get_json(force=True)
    openai_key = resolve_openai_key(data)
    if not openai_key:
        return jsonify({"error": "OpenAI API key required. Enter it in the top-right field or add OPENAI_API_KEY to .env"}), 400
    category = data.get("category", "corporate HR")
    keywords = data.get("keywords", "")

    # Return cached result if same query was run within last 30 minutes (saves ~$0.03/hit)
    cache_key = f"{category}|{keywords}"
    cached = _event_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < 1800:
        return jsonify(cached["data"]), 200

    today       = datetime.now()
    past_date   = today - timedelta(days=7)
    future_date = today + timedelta(days=14)
    date_range  = f"{past_date.strftime('%B %d')} to {future_date.strftime('%B %d, %Y')}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    this_month = today.strftime("%B %Y")
    next_month = (today + timedelta(days=30)).strftime("%B %Y")

    # Event format synonyms — cover every way this category of event is named
    FORMAT_SYNONYMS = "summit conclave conference roundtable forum symposium retreat convention expo meet"

    # ── Step 1: web search for real events ──────────────────────────────
    search_prompt = (
        f"You are a research assistant. Find ALL real {category} events happening INSIDE INDIA ONLY "
        f"between {date_range}. {' Extra focus on: ' + keywords if keywords else ''}\n\n"
        f"CRITICAL RULES:\n"
        f"- ONLY events physically held in Indian cities (Mumbai, Delhi, Bengaluru, Pune, Hyderabad, Chennai, Kolkata, Ahmedabad, Gurugram, Noida, Jaipur, etc).\n"
        f"- DO NOT include events from USA, UK, Singapore, Dubai, or any country outside India.\n"
        f"- Return REAL events with REAL dates — no fabricated events.\n\n"
        f"SEARCH STRATEGY — run these 3 searches (combine sources within each, don't run them one by one):\n\n"
        f"Search 1 — Event aggregators + direct event sites:\n"
        f"  - 10times.com, allevents.in, townscript.com, eventshigh.com: '{category} India {this_month}' and '{category} India {next_month}'\n"
        f"  - '{category} India 2026 registration' — to catch official event sites\n\n"
        f"Search 2 — Broad Google with event format variants:\n"
        f"  - '{category} summit India {this_month}', '{category} conclave India 2026'\n"
        f"  - '{category} conference roundtable forum India {date_range} agenda speakers'\n"
        f"  - Industry media: economictimes.com, hrworld.in, peoplemattersglobal.com, ethrworld.com '{category} event India {date_range}'\n\n"
        f"Search 3 — LinkedIn (events are often ONLY posted here):\n"
        f"  - site:linkedin.com/events '{category}' India 2026\n"
        f"  - site:linkedin.com/posts '{category}' summit OR conclave India {this_month}\n\n"
        f"For EACH event found, provide:\n"
        f"  - Exact event name (as it appears on the source)\n"
        f"  - Exact date or date range\n"
        f"  - Indian city name\n"
        f"  - Status: past (before {today.strftime('%B %d, %Y')}), ongoing (today), or upcoming (after today)\n"
        f"  - One-line description\n"
        f"  - Direct source URL\n\n"
        f"Find as many REAL events as possible. Cast a wide net — {FORMAT_SYNONYMS} all count."
    )

    try:
        search_data = web_search_call(headers, search_prompt, search_context_size="medium")

        # Surface API-level errors (quota, invalid key, model error, etc.)
        try:
            raw_text, annotations = safe_search_extract(search_data)
        except ValueError as ve:
            return jsonify({"error": f"OpenAI search error: {ve}"}), 502

        real_urls = [
            a["url"]
            for a in annotations
            if a.get("type") == "url_citation" and "url" in a
        ]

        # ── Step 2: reformat raw text → clean JSON ───────────────────────
        format_prompt = f"""Convert ALL real events found below into a JSON array. Include every event found — past, ongoing, and upcoming.
Today's date is {today.strftime('%B %d, %Y')}.

Raw event information:
{raw_text}

Real URLs discovered: {json.dumps(real_urls[:12]) if real_urls else '[]'}

Return ONLY a JSON array — no markdown, no extra text:
[{{
  "id": "evt1",
  "name": "Exact event name",
  "date": "Exact date or date range",
  "city": "City, India",
  "status": "past (before today) OR ongoing (today) OR upcoming (after today)",
  "agenda": "One-line description",
  "panels": ["Session/panel topic 1", "Session/panel topic 2"],
  "campaign_topic": "Main outreach theme",
  "reference_url": "Real URL from the list above — must be an actual URL"
}}]

Include every India-based event found. EXCLUDE any event not held in India. Do NOT pad with invented events. Only real events with real URLs."""

        status, fd = gpt_format(headers, format_prompt)
        if status == 200:
            _event_cache[cache_key] = {"ts": time.time(), "data": fd}
        return jsonify(fd), status

    except requests.exceptions.Timeout:
        return jsonify({"error": "Search request timed out — try again"}), 504
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/search-speakers", methods=["POST"])
def search_speakers():
    """Scrape event URL for all speakers + web search supplement, then format to JSON."""
    data       = request.get_json(force=True)
    openai_key = resolve_openai_key(data)
    if not openai_key:
        return jsonify({"error": "OpenAI API key required. Enter it in the top-right field or add OPENAI_API_KEY to .env"}), 400
    event_name = data.get("event_name", "")
    event_city = data.get("event_city", "")
    event_url  = data.get("event_url", "")
    panels     = data.get("panels", [])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    # ── Step 0: Scrape event website directly for speaker list ──
    scraped_text = ""
    if event_url:
        try:
            page_resp = requests.get(event_url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            })
            if page_resp.status_code == 200:
                from html.parser import HTMLParser

                class _TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self._skip = {"script","style","head","meta","link","noscript"}
                        self._cur  = None
                        self.parts = []
                    def handle_starttag(self, tag, attrs): self._cur = tag
                    def handle_data(self, data):
                        if self._cur not in self._skip:
                            t = data.strip()
                            if t: self.parts.append(t)

                ex = _TextExtractor()
                ex.feed(page_resp.text)
                scraped_text = " | ".join(ex.parts)[:10000]
        except Exception:
            pass  # fall through to web search only

    # ── Step 1: Web search to supplement / fill gaps ─────────────
    search_prompt = (
        f'Find ALL speakers and panelists at "{event_name}" in {event_city}, India. '
        f'IMPORTANT: Return EVERY speaker — do not stop at 5 or 10.\n\n'
        f'Search ALL of these sources:\n'
        f'1. Event website directly: {event_url} — extract every name listed as speaker/panelist/moderator\n'
        f'2. LinkedIn: site:linkedin.com "{event_name}" speaker panelist 2026\n'
        f'3. LinkedIn events: site:linkedin.com/events "{event_name}"\n'
        f'4. Google: "{event_name}" all speakers panelists agenda India 2026\n'
        f'5. News: "{event_name}" India speakers hrworld OR ethrworld OR peoplemattersglobal\n\n'
        f'For each speaker: full name, exact job title, company name, '
        f'which session/panel, and source URL.\n'
        f'Panel topics: {", ".join(panels)}.'
    )

    try:
        search_data = web_search_call(headers, search_prompt, search_context_size="low")
        try:
            raw_text, annotations = safe_search_extract(search_data)
        except ValueError as ve:
            return jsonify({"error": f"OpenAI search error: {ve}"}), 502

        source_links = [
            {
                "url":   a["url"],
                "title": a.get("title", a["url"]),
            }
            for a in annotations
            if a.get("type") == "url_citation" and "url" in a
        ]
        real_urls = [s["url"] for s in source_links]

        # ── Step 2: Format ALL speakers → clean JSON ─────────────
        scraped_section = (
            f"\n\nDIRECT WEBSITE SCRAPE (use this — it has ALL speakers):\n{scraped_text}\n"
            if scraped_text else ""
        )

        format_prompt = f"""Extract EVERY speaker from the data below and return as a JSON array.
Event: {event_name}, {event_city}
Panel topics: {", ".join(panels)}

WEB SEARCH RESULTS:
{raw_text}
{scraped_section}
Source URLs: {json.dumps(real_urls[:12]) if real_urls else "[]"}

RULES:
- Include EVERY person listed as speaker / panelist / moderator / keynote.
- Do NOT stop at 5 or 10 — if 30 people are listed, return all 30.
- Do NOT invent people. Only real names from the data above.
- For topic: use the session/panel they appear in, or "Speaker" if unknown.

Return ONLY a JSON array — no markdown, no extra text:
[{{
  "id": "p1",
  "name": "Full name",
  "title": "Exact job title",
  "company": "Exact company name",
  "topic": "Session or panel title they speak on",
  "speaking_about": "What they spoke/will speak about — from real sources",
  "email_hint": "firstname.lastname@company.com",
  "source_url": "URL where this person's name was found"
}}]"""

        # Higher max_tokens because 30+ speakers = much larger JSON
        status, fd = gpt_format(headers, format_prompt, max_tokens=8192)
        fd["source_links"] = source_links[:12]
        return jsonify(fd), status

    except requests.exceptions.Timeout:
        return jsonify({"error": "Speaker search timed out — try again"}), 504
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/search-insights", methods=["POST"])
def search_insights():
    """Search what speakers actually said/are known for, then generate real insights."""
    data       = request.get_json(force=True)
    openai_key = resolve_openai_key(data)
    if not openai_key:
        return jsonify({"error": "OpenAI API key required. Enter it in the top-right field or add OPENAI_API_KEY to .env"}), 400
    event_name = data.get("event_name", "")
    event_city = data.get("event_city", "")
    people     = data.get("people", [])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    speaker_list = "\n".join([
        f"- {p['name']}, {p['title']} at {p['company']} (topic: {p['topic']})"
        for p in people
    ])

    # ── Step 1: Search what each speaker said / is about to say ─
    search_prompt = (
        f'Search for what these speakers said or will say at "{event_name}" in {event_city}:\n'
        f'{speaker_list}\n\n'
        f'For EACH speaker search ALL of these:\n'
        f'1. LinkedIn posts by them: site:linkedin.com "[speaker name]" "{event_name}"\n'
        f'2. LinkedIn posts about the event tagging them: site:linkedin.com "{event_name}" "[speaker name]"\n'
        f'3. Event recap/highlights: "{event_name}" recap highlights key takeaways India\n'
        f'4. If past event: "[speaker name] {event_name} session" OR "{event_name} [speaker name] spoke"\n'
        f'5. If upcoming: "[speaker name] {event_name} speaking" OR "{event_name} agenda [speaker name]"\n'
        f'6. Their recent public views: "[speaker name] [company]" site:linkedin.com OR site:economictimes.com OR site:hrworld.in\n\n'
        f'Determine the event status: past (finished), ongoing (now), or upcoming.'
    )

    try:
        search_data = web_search_call(headers, search_prompt, search_context_size="low")
        try:
            raw_text, annotations = safe_search_extract(search_data)
        except ValueError as ve:
            return jsonify({"error": f"OpenAI search error: {ve}"}), 502

        source_links = [
            {
                "url":   a["url"],
                "title": a.get("title", a["url"]),
            }
            for a in annotations
            if a.get("type") == "url_citation" and "url" in a
        ]
        real_urls = [s["url"] for s in source_links]

        # ── Step 2: Format into per-person insights ─────────────
        people_json = json.dumps([
            {"id": p["id"], "name": p["name"], "title": p["title"],
             "company": p["company"], "topic": p["topic"]}
            for p in people
        ], indent=2)

        format_prompt = f"""Based on REAL web search results, generate personalized outreach insights for these speakers at "{event_name}".

REAL INFORMATION FOUND FROM WEB:
{raw_text}

SPEAKERS:
{people_json}

About Redcliffe Labs:
- India's leading corporate diagnostics & preventive health partner
- Employee health packages: blood tests, screenings, full-body checkups
- Wellness programs proven to reduce absenteeism
- Trusted by 500+ corporates across India

Use the REAL information found. If the event already happened, reference what they actually said. If upcoming/ongoing, reference their known public positions and recent statements.

Return ONLY a JSON array of {len(people)} objects — one per speaker, matched by id:
[{{
  "id": "p1",
  "event_status": "past OR ongoing OR upcoming",
  "speaking_topic": "The exact topic/session title they spoke on, are speaking on, or will speak on at this event — from real sources",
  "what_they_said": "For past: what they actually said or key point made. For ongoing: what they are currently presenting. For upcoming: what they are known to champion or their stated agenda on this topic.",
  "pain_points": ["Real pain point derived from their actual statements or known position", "Second specific pain point"],
  "hook": "One sentence referencing their specific speaking topic or something real they said — not generic",
  "opportunity": "One sentence: exactly how Redcliffe Labs addresses their specific stated concern"
}}]"""

        status, fd = gpt_format(headers, format_prompt, max_tokens=8192)
        fd["source_links"] = source_links[:8]
        return jsonify(fd), status

    except requests.exceptions.Timeout:
        return jsonify({"error": "Insight search timed out — try again"}), 504
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/scrape-event", methods=["POST"])
def scrape_event():
    """Extract event details from a URL — scrapes page content or falls back to web search."""
    data = request.get_json(force=True)
    openai_key = resolve_openai_key(data)
    if not openai_key:
        return jsonify({"error": "OpenAI API key required. Enter it in the top-right field or add OPENAI_API_KEY to .env"}), 400
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400

    headers_ai = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    # ── Try to fetch page content directly ──────────────────────
    page_text = ""
    try:
        page_resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        })
        if page_resp.status_code == 200:
            from html.parser import HTMLParser

            class _TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._skip = {"script","style","head","meta","link","noscript"}
                    self._cur  = None
                    self.parts = []
                def handle_starttag(self, tag, attrs): self._cur = tag
                def handle_data(self, data):
                    if self._cur not in self._skip:
                        t = data.strip()
                        if t: self.parts.append(t)

            ex = _TextExtractor()
            ex.feed(page_resp.text)
            page_text = " ".join(ex.parts)[:6000]
    except Exception:
        pass  # fall through to web search

    today = datetime.now().strftime("%B %d, %Y")

    if page_text:
        # ── Extract from scraped page ──────────────────────────
        extract_prompt = f"""Today is {today}. Extract the India-based event details from this webpage.

URL: {url}
Page content:
{page_text}

Return as {{\"result\": [{{...}}]}} with exactly one event:
{{
  "id": "custom1",
  "name": "Exact event name",
  "date": "Exact event date or date range",
  "city": "Indian city name only",
  "status": "past OR ongoing OR upcoming (relative to {today})",
  "agenda": "One-line event description",
  "panels": ["Session or panel topic 1", "Session or panel topic 2"],
  "campaign_topic": "Main outreach theme for B2B sales",
  "reference_url": "{url}"
}}"""
        status, fd = gpt_format(headers_ai, extract_prompt)
        return jsonify(fd), status

    else:
        # ── Fallback: web search for this specific event URL ───
        search_prompt = (
            f"Search the web for details about this India-based event: {url} . "
            f"Find the event name, exact date, city in India, agenda, panel topics, and speaker information. "
            f"Today is {today}."
        )
        try:
            search_data = web_search_call(headers_ai, search_prompt, search_context_size="low")
            try:
                raw_text, _ = safe_search_extract(search_data)
            except ValueError as ve:
                return jsonify({"error": f"OpenAI search error: {ve}"}), 502

            format_prompt = f"""Today is {today}. Extract the India-based event details from this web search result.

Search result about: {url}
{raw_text}

Return as {{\"result\": [{{...}}]}} with exactly one event:
{{
  "id": "custom1",
  "name": "Exact event name",
  "date": "Exact date or date range",
  "city": "Indian city name only",
  "status": "past OR ongoing OR upcoming (relative to {today})",
  "agenda": "One-line description",
  "panels": ["Session or panel topic 1", "Session or panel topic 2"],
  "campaign_topic": "Main B2B outreach theme",
  "reference_url": "{url}"
}}"""
            status, fd = gpt_format(headers_ai, format_prompt)
            return jsonify(fd), status

        except requests.exceptions.Timeout:
            return jsonify({"error": "Request timed out — try again"}), 504
        except requests.exceptions.RequestException as exc:
            return jsonify({"error": str(exc)}), 502


@app.route("/api/event-recap", methods=["POST"])
def event_recap():
    """Search for what actually happened at a past event — real recap, highlights, outcomes."""
    data       = request.get_json(force=True)
    openai_key = resolve_openai_key(data)
    if not openai_key:
        return jsonify({"error": "OpenAI API key required. Enter it in the top-right field or add OPENAI_API_KEY to .env"}), 400
    event_name = data.get("event_name", "")
    event_city = data.get("event_city", "")
    event_date = data.get("event_date", "")
    event_url  = data.get("event_url", "")

    headers_ai = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    search_prompt = (
        f'Find real recap, highlights and outcomes of "{event_name}" held in {event_city} on {event_date}. '
        f'Search ALL of these:\n'
        f'1. LinkedIn posts tagged #{event_name.replace(" ","")} OR mentioning "{event_name}" recap highlights\n'
        f'2. site:linkedin.com/posts "{event_name}" recap OR highlights OR "key takeaways"\n'
        f'3. News: "{event_name}" recap OR highlights OR summary — on economictimes.com, hrworld.in, '
        f'businessline.com, peoplemattersglobal.com, ethrworld.com\n'
        f'4. Google: "{event_name}" {event_city} {event_date} recap highlights what happened\n'
        f'{("5. Event website: " + event_url) if event_url else ""}\n\n'
        f'Find: (a) Was the event actually held? '
        f'(b) Key themes and topics discussed '
        f'(c) Notable speakers or announcements '
        f'(d) Any photos, posts, or news coverage confirming it happened '
        f'(e) Attendee count or scale if mentioned. '
        f'Return real facts only — no invented content.'
    )

    try:
        search_data = web_search_call(headers_ai, search_prompt, search_context_size="low")
        try:
            raw_text, annotations = safe_search_extract(search_data)
        except ValueError as ve:
            return jsonify({"error": f"OpenAI search error: {ve}"}), 502

        source_links = [
            {"url": a["url"], "title": a.get("title", "")}
            for a in annotations
            if a.get("type") == "url_citation" and "url" in a
        ]

        status_code, fd = gpt_format(headers_ai, f"""Summarise what happened at this event based on real web search results.

Event: {event_name}, {event_city}, {event_date}

Web search results:
{raw_text}

Return as {{"result": [{{...}}]}} with exactly one object:
{{
  "confirmed": true or false (was the event actually confirmed to have happened?),
  "confirmation_note": "One sentence — what confirmed it happened (e.g. LinkedIn posts, news article, photos)",
  "key_themes": ["Theme 1 actually discussed", "Theme 2", "Theme 3"],
  "highlights": "2-3 sentences on the main discussions, announcements, or outcomes from the event",
  "coverage_links": ["URL1", "URL2"]
}}""")

        fd["source_links"] = source_links[:6]
        return jsonify(fd), status_code

    except requests.exceptions.Timeout:
        return jsonify({"error": "Recap search timed out — try again"}), 504
    except requests.exceptions.RequestException as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/find-emails", methods=["POST"])
def find_emails():
    """Look up verified emails via Apollo.ai for selected people only."""
    data       = request.get_json(force=True)
    apollo_key = (data.get("apollo_key") or "").strip() or os.environ.get("APOLLO_API_KEY", "")
    people     = data.get("people", [])   # only checked people passed from frontend

    if not apollo_key:
        return jsonify({"error": "Apollo API key required. Enter it in the form or add APOLLO_API_KEY to .env"}), 400
    if not people:
        return jsonify({"results": {}}), 200

    results = {}
    for person in people:
        pid    = person.get("id", "")
        parts  = person.get("name", "").strip().split()
        first  = parts[0] if parts else ""
        last   = " ".join(parts[1:]) if len(parts) > 1 else ""

        try:
            resp = requests.post(
                "https://api.apollo.io/v1/people/match",
                headers={
                    "Content-Type":  "application/json",
                    "Cache-Control": "no-cache",
                    "x-api-key":     apollo_key,
                },
                json={
                    "api_key":                apollo_key,
                    "first_name":             first,
                    "last_name":              last,
                    "organization_name":      person.get("company", ""),
                    "reveal_personal_emails": True,   # spend credits to get the real email
                    "reveal_phone_number":    False,
                },
                timeout=15,
            )
            raw    = resp.json()
            pr     = raw.get("person") or {}
            email  = pr.get("email") or ""
            status = pr.get("email_status") or ("guessed" if pr else "not_found")
            if not email and pr.get("email_addresses"):
                for ea in pr["email_addresses"]:
                    if ea.get("email"):
                        email  = ea["email"]
                        status = ea.get("email_status", "likely")
                        break
            results[pid] = {"email": email, "status": status, "found": bool(email)}
        except Exception as exc:
            results[pid] = {"email": "", "status": "error", "found": False, "error": str(exc)}

        time.sleep(0.25)   # Apollo free tier: avoid hitting rate limits

    return jsonify({"results": results})


@app.route("/api/preview-bulk", methods=["POST"])
def preview_bulk():
    """Generate personalised email content for all selected speakers — no sending."""
    data        = request.get_json(force=True)
    openai_key  = resolve_openai_key(data)
    if not openai_key:
        return jsonify({"error": "OpenAI API key required. Enter it in the top-right field or add OPENAI_API_KEY to .env"}), 400
    event       = data.get("event", {})
    people      = data.get("people", [])
    insights    = data.get("insights", [])
    sender_name = data.get("sender_name", "Redcliffe Labs").strip()
    sender_role = data.get("sender_role", "Corporate Partnerships").strip()
    pitch_angle = data.get("pitch_angle", "I heard your panel session")

    insight_map = {ins["id"]: ins for ins in insights}
    headers_ai  = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {openai_key}"}

    previews = {}
    for person in people:
        pid     = person.get("id", "")
        insight = insight_map.get(pid, {})

        prompt = (
            f"Write a warm B2B outreach email from {sender_name}, {sender_role} at Redcliffe Labs.\n"
            f"Recipient: {person.get('name')}, {person.get('title')} at {person.get('company')}\n"
            f"Event: {event.get('name')}, {event.get('city')}\n"
            f"Their session: {person.get('topic','')}\n"
            f"Pitch angle: {pitch_angle}\n"
            f"Hook: {insight.get('hook','')}\n"
            f"Key opportunity: {insight.get('opportunity','')}\n\n"
            f"Rules: body under 120 words, warm genuine tone, reference event + session, "
            f"15-min CTA, sign: {sender_name}, {sender_role}, Redcliffe Labs.\n"
            f'Return JSON: {{"subject": "...", "body": "..."}}'
        )
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers_ai,
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 400,
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "Return JSON with subject and body only."},
                        {"role": "user",   "content": prompt},
                    ],
                },
                timeout=30,
            )
            content = json.loads(resp.json()["choices"][0]["message"]["content"])
            previews[pid] = {
                "subject": content.get("subject", ""),
                "body":    content.get("body", ""),
            }
        except Exception:
            previews[pid] = {"subject": f"Partnership — {event.get('name','')}", "body": ""}

    return jsonify({"previews": previews})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, use_reloader=False, threaded=True)
