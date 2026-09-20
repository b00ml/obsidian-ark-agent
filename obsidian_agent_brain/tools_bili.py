"""B站视频工具：薄封装 bili_transcript.py（3层降级字幕 / 元数据 / 多模态）。

复用既有函数，零重写；通过 chdir 到 bili_summarizer 保持临时文件行为一致，
完成后清理临时产物。敏感信息（Cookie）仅本机读取，不返回内容。
"""
import os
import re
import uuid

from common import chdir, setup_paths, vault_root


def _stage_tracker(config: dict, pipeline: str):
    """Create one request-local, serializable stage trace when agentlab is available."""
    setup_paths(config)
    try:
        from agentlab.contracts import ProcessStatus
        from agentlab.runtime.operation_context import current_operation_id
        from agentlab.runtime.stages import StageTracker
    except (ImportError, ModuleNotFoundError):
        return None, None
    operation_id = current_operation_id()
    run_id = operation_id or f"{pipeline}-{uuid.uuid4().hex[:16]}"
    return StageTracker(run_id, uuid.uuid4().hex[:16], operation_id=operation_id), ProcessStatus


def _record_stage(tracker, statuses, stage_id: str, *, status, warnings=None,
                  artifact_refs=None, error_code: str = "", retryable: bool = False) -> None:
    if tracker is not None:
        tracker.record(stage_id, stage_id, status=status, warnings=warnings,
                       artifact_refs=artifact_refs, error_code=error_code,
                       retryable=retryable)


def _with_stages(result: dict, tracker) -> dict:
    if tracker is not None:
        result["stages"] = tracker.to_dict()
    return result


def _import_bili(config: dict):
    setup_paths(config)
    import bili_transcript  # sys.path 已加入 bili_summarizer
    return bili_transcript


def _cookie_path(config: dict) -> str:
    cp = config.get("cookie_path", "bili_summarizer/bilibili_cookie.json")
    if not os.path.isabs(cp):
        cp = os.path.join(config.get("project_root", ""), cp)
    return cp


def _visual_path(config: dict) -> str:
    """visual_models.json 绝对路径（相对 project_root 解析）。

    _run_in_bili 会 chdir 到 bili_summarizer，相对路径在此 cwd 下会再拼一段而成
    bili_summarizer/bili_summarizer/... 导致 FileNotFoundError，故预先转绝对路径。
    """
    vp = config.get("visual_models_path", "config/visual_models.json")
    if not os.path.isabs(vp):
        vp = os.path.join(config.get("project_root", ""), vp)
    return vp


def _transcript_note(config: dict, bvid: str, title: str,
                     transcript: str) -> str | None:
    """把完整逐字稿落盘为 Inbox/{safe_title}-逐字稿.md。

    transcript 在工具内直接落盘、不经 agent，保证完整不截断。与渲染笔记
    共用 safe_title（标题清理非法字符）命名，便于总结侧挂 [[{safe}-逐字稿]]。
    """
    if not transcript:
        return None
    safe = re.sub(r'[\\/:*?"<>|]', "_", title or bvid)
    note_dir = os.path.join(vault_root(config) or "", "Inbox")
    os.makedirs(note_dir, exist_ok=True)
    head = [
        "---",
        f'title: "{title or bvid} - 逐字稿"',
        "type: video-transcript",
        f'source: "https://www.bilibili.com/video/{bvid}"',
        "tags: [video, transcript]",
        "---",
        "",
        f"> 视频总结见 [[{safe}-总结]]",
        "",
    ]
    path = os.path.join(note_dir, f"{safe}-逐字稿.md")
    _pipeline_write(path, "\n".join(head) + "\n\n" + str(transcript).strip() + "\n",
                    config, actor="brain-bili-transcript")
    return path


def _pipeline_write(path: str, content: str, config: dict, actor: str) -> dict:
    """brain 内管线产物的受控写入：复用 bili_summarizer.vault_io（P0-01）。"""
    import sys as _sys
    _bili_dir = os.path.join(config.get("project_root", ""), "bili_summarizer")
    if _bili_dir not in _sys.path:
        _sys.path.insert(0, _bili_dir)
    from vault_io import controlled_write
    return controlled_write(path, content, vault_root=vault_root(config),
                            overwrite=True, actor=actor)


def _cleanup_temp(workdir: str) -> list[str]:
    removed = []
    for name in os.listdir(workdir):
        if (name.startswith("temp_audio") or name.startswith("temp_subtitle")
                or name.startswith("temp_video") or name.startswith("transcript_")
                or name.startswith("meta_")):
            p = os.path.join(workdir, name)
            try:
                os.remove(p)
                removed.append(name)
            except OSError:
                pass
    frames = os.path.join(workdir, "frames_tmp")
    if os.path.isdir(frames):
        try:
            for f in os.listdir(frames):
                os.remove(os.path.join(frames, f))
            os.rmdir(frames)
        except OSError:
            pass
    return removed


def _run_in_bili(func, config: dict, *args, **kwargs):
    """在 bili_summarizer 工作目录执行既有函数并清理临时文件"""
    workdir = os.path.join(config.get("project_root", ""), "bili_summarizer")
    if not os.path.isdir(workdir):
        workdir = "."
    with chdir(workdir):
        try:
            return func(*args, **kwargs)
        finally:
            _cleanup_temp(workdir)


def bili_transcribe(config: dict, bvid: str, quality: str = "fast",
                    trust_ai: bool = False,
                    whisper_model: str | None = None,
                    whisper_device: str | None = None,
                    beam: int | None = None,
                    vad: bool | None = None) -> dict:
    """提取 B站字幕（3层降级：B站API → yt-dlp → Whisper）。

    whisper_model/whisper_device/beam/vad 透传给策略3（Whisper），
    为 None 时按其默认规则自动决定（GPU 加速、模型映射、VAD 开启）。
    返回 title / transcript / strategy / meta；全部失败抛错（含降级链路信息）。
    """
    tracker, statuses = _stage_tracker(config, "bili-transcribe")
    try:
        _record_stage(tracker, statuses, "accepted", status=statuses.ACCEPTED)
        bili = _import_bili(config)
        bvid = bili.extract_bvid(bvid)
        cookie_path = _cookie_path(config)

        def _run():
            bili.ensure_ffmpeg()
            cookie = bili.load_cookie(cookie_path)
            result = bili.strategy_api(bvid, cookie, trust_ai=trust_ai)
            strategy = 1
            if not result and cookie:
                new_cookie = bili.refresh_cookie(cookie, cookie_path)
                if new_cookie != cookie:
                    cookie = new_cookie
                    result = bili.strategy_api(bvid, cookie, trust_ai=trust_ai)
            if not result:
                result = bili.strategy_ytdlp_subtitle(bvid)
                strategy = 2
            if not result:
                result = bili.strategy_whisper(
                    bvid, quality,
                    whisper_model=whisper_model,
                    device=whisper_device,
                    beam=beam,
                    vad=vad,
                )
                strategy = 3
            if not result:
                raise RuntimeError(
                    f"[BILI] 3层降级全部失败 bvid={bvid} "
                    f"(API → yt-dlp → Whisper)，请检查 Cookie/网络/BV号")
            title, transcript = result
            return title, transcript, strategy, cookie

        title, transcript, strategy, cookie = _run_in_bili(_run, config)
        _record_stage(tracker, statuses, "fetched", status=statuses.FETCHED)
        _record_stage(tracker, statuses, "parsed", status=statuses.PARSED)
        meta = _run_in_bili(lambda: bili.get_video_meta(bvid, cookie), config)
        _record_stage(tracker, statuses, "enriched", status=statuses.ENRICHED)
        _record_stage(tracker, statuses, "completed", status=statuses.COMPLETED)
        return _with_stages({"bvid": bvid, "title": title, "strategy": strategy,
                             "char_count": len(transcript), "transcript": transcript,
                             "meta": meta}, tracker)
    except Exception as exc:
        if statuses is not None:
            _record_stage(tracker, statuses, "failed", status=statuses.FAILED,
                          error_code=type(exc).__name__)
        raise


def bili_meta(config: dict, bvid: str) -> dict:
    """获取 B站视频元数据（标题/作者/日期/时长），失败返回最小 meta。"""
    bili = _import_bili(config)
    bvid = bili.extract_bvid(bvid)
    cookie_path = _cookie_path(config)

    def _run():
        return bili.get_video_meta(bvid, bili.load_cookie(cookie_path))

    return _run_in_bili(_run, config)


def _prepare_visual(config: dict, bvid: str, transcript: str):
    """多模态前置：确保字幕 + 标题 + Cookie + 视觉配置就绪"""
    bili = _import_bili(config)
    bvid = bili.extract_bvid(bvid)
    cookie_path = _cookie_path(config)
    if not transcript:
        res = bili_transcribe(config, bvid)
        transcript = res["transcript"]
    title = bili.get_title_via_ytdlp(bvid) or bvid
    # 用 get_video_meta（与 bili_meta 同源）优先拿真实内容标题，
    # 避免 yt-dlp 取标题失败时回退成 BV 号，导致笔记文件名/标题变成 BV 命名
    try:
        cookie = bili.load_cookie(cookie_path)
        real_title = bili.get_video_meta(bvid, cookie).get("title")
        if real_title:
            title = real_title
    except Exception:
        pass
    return bili, bvid, title, transcript, cookie_path


def bili_screenshot(config: dict, bvid: str, transcript: str = "") -> dict:
    """模式A：LLM 决定关键截图时间点 → 截图 → 生成截图笔记到 Inbox/。

    需要 visual_models.json 配置 API Key；缺 Key 或依赖时报错并提示降级。
    """
    bili, bvid, title, transcript, cookie_path = _prepare_visual(config, bvid, transcript)

    def _run():
        import visual_analyzer as va
        full_config = va.load_visual_config(_visual_path(config))
        model_config = full_config.get("default", {})
        screenshot_cfg = full_config.get("screenshot", {})
        cookie = bili.load_cookie(cookie_path)
        note = bili.run_screenshot_mode(bvid, title, transcript, model_config,
                                        screenshot_cfg,
                                        vault_path=vault_root(config), cookie=cookie)
        return note

    note = _run_in_bili(_run, config)
    tp = _transcript_note(config, bvid, title, transcript)
    msg = f"截图笔记: {note}"
    if tp:
        msg += f"；逐字稿: {tp}"
    return {"bvid": bvid, "note": note, "transcript_note": tp,
            "status": "ok" if note else "no_markers",
            "message": msg if note else "未生成截图标记，跳过"}


def bili_visual(config: dict, bvid: str, transcript: str = "") -> dict:
    """模式B：截帧 → 网格图 → 视觉模型分析 → 增强笔记到 Inbox/。

    需要 visual_models.json 配置视觉模型 API Key。
    """
    bili, bvid, title, transcript, cookie_path = _prepare_visual(config, bvid, transcript)

    def _run():
        import visual_analyzer as va
        full_config = va.load_visual_config(_visual_path(config))
        visual_config = full_config.get("visual", full_config.get("default", {}))
        grid_cfg = full_config.get("grid", {})
        cookie = bili.load_cookie(cookie_path)
        note = bili.run_visual_mode(bvid, title, transcript, visual_config, grid_cfg,
                                    vault_path=vault_root(config), cookie=cookie)
        return note

    note = _run_in_bili(_run, config)
    tp = _transcript_note(config, bvid, title, transcript)
    msg = f"视觉分析笔记: {note}"
    if tp:
        msg += f"；逐字稿: {tp}"
    return {"bvid": bvid, "note": note, "transcript_note": tp,
            "status": "ok" if note else "failed",
            "message": msg if note else "截帧/视觉分析失败，已降级"}
