#!/usr/bin/env python
"""
IndexTTS2 API 测试脚本
用法: python test_api.py [--host HOST] [--port PORT]

注意: 需要先启动 API 服务器:
  python api_server.py --host 0.0.0.0 --port 8002 --fp16
"""
import argparse
import requests
import json
import sys
import time
import os


BASE_URL = "http://localhost:8002"


def test_health():
    """测试健康检查"""
    print("\n[1] 测试 /health...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"    错误: 无法连接到服务器")
        return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_get_voices():
    """测试获取角色列表"""
    print("\n[2] 测试 /audio/voices...")
    try:
        r = requests.get(f"{BASE_URL}/audio/voices", timeout=5)
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_register_speaker(name="测试角色", audio_path="examples/voice_01.wav"):
    """测试注册角色"""
    print(f"\n[3] 测试 /register_speaker ({name})...")
    try:
        r = requests.post(
            f"{BASE_URL}/register_speaker",
            json={"name": name, "sample_audios": [audio_path]},
            timeout=10
        )
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_tts_wav(text="你好，这是测试", voice="examples/voice_01.wav"):
    """测试 TTS 合成"""
    print(f"\n[4] 测试 /tts-wav...")
    try:
        r = requests.post(
            f"{BASE_URL}/tts-wav",
            json={"text": text, "spk_audio_prompt": voice},
            timeout=120
        )
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
            # 保存音频文件
            output_file = f"outputs/test_{int(time.time())}.wav"
            os.makedirs("outputs", exist_ok=True)
            with open(output_file, "wb") as f:
                f.write(r.content)
            print(f"    已保存: {output_file} ({len(r.content)} bytes)")
            return True
        else:
            print(f"    错误: {r.json()}")
            return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_tts(text="你好", voice="examples/voice_01.wav"):
    """测试 TTS 二进制"""
    print(f"\n[5] 测试 /tts...")
    try:
        r = requests.post(
            f"{BASE_URL}/tts",
            json={"text": text, "spk_audio_prompt": voice},
            timeout=120
        )
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
            print(f"    返回二进制大小: {len(r.content)} bytes")
            return True
        else:
            print(f"    错误: {r.json()}")
            return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_tts_url(text="测试", audio_path="examples/voice_01.wav"):
    """测试 /tts_url"""
    print(f"\n[6] 测试 /tts_url...")
    try:
        r = requests.post(
            f"{BASE_URL}/tts_url",
            json={"text": text, "audio_paths": [audio_path]},
            timeout=120
        )
        print(f"    状态码: {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_openai_compat(text="测试", voice="examples/voice_01.wav"):
    """测试 OpenAI 兼容接口"""
    print(f"\n[7] 测试 /audio/speech...")
    try:
        r = requests.post(
            f"{BASE_URL}/audio/speech",
            json={"input": text, "voice": voice},
            timeout=120
        )
        print(f"    状态码: {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_upload_audio(file_path):
    """测试文件上传"""
    print(f"\n[8] 测试 /upload_audio...")
    try:
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/upload_audio",
                files={"file": f},
                timeout=30
            )
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_update_speaker(name="测试角色", audio_path="examples/voice_01.wav"):
    """测试更新角色"""
    print(f"\n[9] 测试 /update_speaker...")
    try:
        r = requests.post(
            f"{BASE_URL}/update_speaker",
            json={"name": name, "sample_audios": [audio_path]},
            timeout=10
        )
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_delete_speaker(name="测试角色"):
    """测试删除角色"""
    print(f"\n[10] 测试 /delete_speaker...")
    try:
        r = requests.post(
            f"{BASE_URL}/delete_speaker",
            json={"name": name},
            timeout=10
        )
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 API 测试")
    parser.add_argument("--host", default="localhost", help="API 主机")
    parser.add_argument("--port", type=int, default=8002, help="API 端口")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://{args.host}:{args.port}"

    print("=" * 50)
    print("IndexTTS2 API 测试")
    print("=" * 50)

    # 先检查健康状态
    if not test_health():
        print("\n❌ API 服务未启动!")
        print("\n请先启动 API 服务器:")
        print("  cd C:\\Users\\wushuo\\Desktop\\indextts\\index-tts-official")
        print("  python api_server.py --fp16")
        print("\n然后在另一个终端运行:")
        print("  python test_api.py")
        sys.exit(1)

    results = []

    # 测试所���接��
    results.append(("获取角色列表", test_get_voices()))
    results.append(("注册角色", test_register_speaker()))
    results.append(("TTS WAV", test_tts_wav()))
    results.append(("TTS 二进制", test_tts()))
    results.append(("TTS URL", test_tts_url()))
    results.append(("OpenAI兼容", test_openai_compat()))
    results.append(("更新角色", test_update_speaker()))
    results.append(("删除角色", test_delete_speaker()))

    # 总结
    print("\n" + "=" * 50)
    print("测试结果:")
    print("=" * 50)
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{len(results)} 通过")

    if passed == len(results):
        print("\n🎉 所有接口测试通过!")
    else:
        print("\n⚠️ 部分接口测试失败")


if __name__ == "__main__":
    main()