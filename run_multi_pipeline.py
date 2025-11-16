#!/usr/bin/env python3
"""
멀티 스트림 파이프라인 실행 스크립트
여러 RTSP 스트림을 동시에 처리합니다.
"""
import logging
import multiprocessing
import os
import signal
import sys
from typing import List

from app.config import Settings
from app.runner import create_pipeline


# 전역 변수로 프로세스 리스트 관리
processes: List[multiprocessing.Process] = []


def signal_handler(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM"""
    logging.info(f"Received signal {signum}, shutting down all pipelines...")
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    sys.exit(0)


def run_single_pipeline(stream_id: str, rtsp_url: str, base_settings: dict):
    """단일 스트림을 위한 파이프라인 실행"""
    # 각 프로세스마다 독립적인 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{stream_id}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting pipeline for stream {stream_id}: {rtsp_url}")

    try:
        # 설정 오버라이드
        settings_dict = base_settings.copy()
        settings_dict['media_rpi_rtsp_url'] = rtsp_url
        settings_dict['scarecrow_serial_number'] = stream_id

        settings = Settings(**settings_dict)
        pipeline = create_pipeline(settings)

        logger.info(f"Pipeline for {stream_id} initialized successfully")
        pipeline.run_forever()

    except KeyboardInterrupt:
        logger.info(f"Pipeline {stream_id} interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline {stream_id} failed: {e}", exc_info=True)
        raise


def main():
    """멀티 스트림 파이프라인 메인 함수"""
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 기본 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='[MAIN] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 기본 설정 로드 (.env 파일)
    base_settings_obj = Settings()

    # Settings를 dict로 변환
    base_settings = {
        field: getattr(base_settings_obj, field)
        for field in base_settings_obj.model_fields.keys()
    }

    # 환경 변수에서 스트림 리스트 읽기
    # RTSP_STREAMS=00000000,99999999 형식
    stream_ids = os.getenv('RTSP_STREAMS', '00000000,99999999').split(',')
    stream_ids = [s.strip() for s in stream_ids if s.strip()]

    # RTSP 베이스 URL (환경 변수 또는 기본값)
    rtsp_base_url = os.getenv('RTSP_BASE_URL', 'rtsp://k13e106.p.ssafy.io:8554/stream')

    logger.info(f"Starting multi-stream pipeline for {len(stream_ids)} streams: {stream_ids}")
    logger.info(f"RTSP base URL: {rtsp_base_url}")

    # 각 스트림에 대해 별도 프로세스 생성
    for stream_id in stream_ids:
        rtsp_url = f"{rtsp_base_url}/{stream_id}"

        proc = multiprocessing.Process(
            target=run_single_pipeline,
            args=(stream_id, rtsp_url, base_settings),
            name=f"pipeline-{stream_id}"
        )
        processes.append(proc)
        proc.start()
        logger.info(f"Started process for stream {stream_id} (PID: {proc.pid})")

    # 모든 프로세스가 종료될 때까지 대기
    logger.info(f"All {len(processes)} pipeline processes started. Waiting for completion...")

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, terminating all processes...")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

    logger.info("All pipeline processes have terminated.")


if __name__ == '__main__':
    # multiprocessing에서 필요
    multiprocessing.set_start_method('spawn', force=True)
    main()
