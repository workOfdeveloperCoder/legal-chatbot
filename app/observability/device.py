from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device_type: str
    device_os: str | None
    device_browser: str | None


def parse_device_info(user_agent: str | None) -> DeviceInfo:
    if not user_agent:
        return DeviceInfo(
            device_type="unknown",
            device_os=None,
            device_browser=None,
        )

    ua = user_agent.lower()

    if any(bot in ua for bot in ("bot", "crawler", "spider", "curl", "postman")):
        device_type = "bot"
    elif any(mobile in ua for mobile in ("iphone", "android", "mobile")):
        device_type = "mobile"
    elif "ipad" in ua or "tablet" in ua:
        device_type = "tablet"
    else:
        device_type = "desktop"

    os_name = None
    if "windows" in ua:
        os_name = "Windows"
    elif "mac os" in ua or "macintosh" in ua:
        os_name = "macOS"
    elif "android" in ua:
        os_name = "Android"
    elif "iphone" in ua or "ios" in ua:
        os_name = "iOS"
    elif "linux" in ua:
        os_name = "Linux"

    browser = None
    browser_patterns = (
        ("Edg/", "Edge"),
        ("Chrome/", "Chrome"),
        ("Firefox/", "Firefox"),
        ("Safari/", "Safari"),
        ("OPR/", "Opera"),
    )

    for marker, name in browser_patterns:
        if marker.lower() in ua:
            browser = name
            break

    if browser is None:
        match = re.search(r"([a-zA-Z]+)/[\d.]+", user_agent)
        if match:
            browser = match.group(1)

    return DeviceInfo(
        device_type=device_type,
        device_os=os_name,
        device_browser=browser,
    )
