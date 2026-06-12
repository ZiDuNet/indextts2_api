#!/usr/bin/env python
"""
IndexTTS2 API 测试脚本
用法: python test_api.py [--host HOST] [--port PORT]

需要先启动 API 服务器:
  uv run python api_server.py --host 0.0.0.0 --port 8002 --fp16
"""
import argparse
import requests
import sys
import time
import os


BASE_URL = "http://localhost:8002"
VOICE_ID = None


def test_health():
    """测试健康检查"""
    print("\n[1] 测试 GET /health...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        print("    错误: 无法连接到服务器")
        return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_list_voices():
    """测试获取音色列表"""
    print("\n[2] 测试 GET /v1/audio/voices...")
    try:
        r = requests.get(f"{BASE_URL}/v1/audio/voices", timeout=5)
        print(f"    状态码: {r.status_code}")
        data = r.json()
        print(f"    已注册音色: {len(data.get('data', []))} 个")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_upload_voice():
    """测试上传音色"""
    global VOICE_ID
    print("\n[3] 测试 POST /v1/audio/voices...")
    audio_path = "examples/voice_01.wav"
    if not os.path.exists(audio_path):
        print(f"    跳过: {audio_path} 不存在")
        return None

    try:
        with open(audio_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/v1/audio/voices",
                files={"audio": ("voice_01.wav", f, "audio/wav")},
                data={"speaker_name": "测试音色"},
                timeout=120,
            )
        print(f"    状态码: {r.status_code}")
        result = r.json()
        print(f"    响应: {result}")
        if r.status_code == 200:
            VOICE_ID = result.get("voice_id")
            print(f"    voice_id: {VOICE_ID}")
            return True
        return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_speech():
    """测试语音合成"""
    print("\n[4] 测试 POST /v1/audio/speech...")
    try:
        r = requests.post(
            f"{BASE_URL}/v1/audio/speech",
            json={"input": "你好，这是一个测试", "voice": VOICE_ID or "examples/voice_01.wav"},
            timeout=120,
        )
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
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


def test_speech_emotion():
    """测试情感合成"""
    print("\n[5] 测试 POST /v1/audio/speech (情感向量)...")
    try:
        r = requests.post(
            f"{BASE_URL}/v1/audio/speech",
            json={
                "input": "对不起嘛！我的记性真的不太好",
                "voice": VOICE_ID or "examples/voice_01.wav",
                "emo_vector": [0, 0, 0.8, 0, 0, 0, 0, 0],
            },
            timeout=120,
        )
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
            print(f"    音频大小: {len(r.content)} bytes")
            return True
        else:
            print(f"    错误: {r.json()}")
            return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_speech_mp3():
    """测试 MP3 输出"""
    print("\n[6] 测试 POST /v1/audio/speech (MP3 格式)...")
    try:
        r = requests.post(
            f"{BASE_URL}/v1/audio/speech",
            json={
                "input": "MP3格式测试",
                "voice": VOICE_ID or "examples/voice_01.wav",
                "response_format": "mp3",
            },
            timeout=120,
        )
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
            print(f"    Content-Type: {r.headers.get('Content-Type')}")
            print(f"    音频大小: {len(r.content)} bytes")
            return True
        else:
            print(f"    错误: {r.json()}")
            return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_speech_emo_text():
    """测试文本情感"""
    print("\n[7] 测试 POST /v1/audio/speech (文本情感)...")
    try:
        r = requests.post(
            f"{BASE_URL}/v1/audio/speech",
            json={
                "input": "快躲起来！是他要来了！",
                "voice": VOICE_ID or "examples/voice_01.wav",
                "use_emo_text": True,
                "emo_alpha": 0.6,
            },
            timeout=120,
        )
        print(f"    状态码: {r.status_code}")
        if r.status_code == 200:
            print(f"    音频大小: {len(r.content)} bytes")
            return True
        else:
            print(f"    错误: {r.json()}")
            return False
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_delete_voice():
    """测试删除音色"""
    if not VOICE_ID:
        print("\n[8] 测试 DELETE /v1/audio/voices/{id}... 跳过（未注册音色）")
        return None

    print(f"\n[8] 测试 DELETE /v1/audio/voices/{VOICE_ID}...")
    try:
        r = requests.delete(f"{BASE_URL}/v1/audio/voices/{VOICE_ID}", timeout=10)
        print(f"    状态码: {r.status_code}")
        print(f"    响应: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"    错误: {e}")
        return False


def test_error_cases():
    """测试错误场景"""
    print("\n[9] 测试异常场景...")
    passed = 0

    # 缺少 input
    r = requests.post(f"{BASE_URL}/v1/audio/speech", json={}, timeout=10)
    if r.status_code == 400:
        print("    缺少 input → 400 ✅")
        passed += 1
    else:
        print(f"    缺少 input → {r.status_code} ❌")

    # 不存在的音色
    r = requests.post(f"{BASE_URL}/v1/audio/speech",
                      json={"input": "测试", "voice": "nonexistent_voice"}, timeout=10)
    if r.status_code == 400:
        print("    不存在的音色 → 400 ✅")
        passed += 1
    else:
        print(f"    不存在的音色 → {r.status_code} ❌")

    # 不存在的 voice_id 删除
    r = requests.delete(f"{BASE_URL}/v1/audio/voices/nonexistent", timeout=10)
    if r.status_code == 404:
        print("    删除不存在的音色 → 404 ✅")
        passed += 1
    else:
        print(f"    删除不存在的音色 → {r.status_code} ❌")

    # 错误的 emo_vector
    r = requests.post(f"{BASE_URL}/v1/audio/speech",
                      json={"input": "测试", "emo_vector": [1, 2, 3]}, timeout=10)
    if r.status_code == 400:
        print("    emo_vector 长度错误 → 400 ✅")
        passed += 1
    else:
        print(f"    emo_vector 长度错误 → {r.status_code} ❌")

    return passed == 4


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

    if not test_health():
        print("\nAPI 服务未启动!")
        print("请先启动: uv run python api_server.py --fp16")
        sys.exit(1)

    results = []
    results.append(("获取音色列表", test_list_voices()))
    results.append(("上传音色", test_upload_voice()))
    results.append(("语音合成", test_speech()))
    results.append(("情感合成", test_speech_emotion()))
    results.append(("MP3 输出", test_speech_mp3()))
    results.append(("文本情感", test_speech_emo_text()))
    results.append(("删除音色", test_delete_voice()))
    results.append(("异常场景", test_error_cases()))

    print("\n" + "=" * 50)
    print("测试结果:")
    print("=" * 50)
    passed = 0
    for name, result in results:
        if result is None:
            status = "⏭ 跳过"
        elif result:
            status = "✅ 通过"
            passed += 1
        else:
            status = "❌ 失败"
        print(f"  {name}: {status}")

    total = len([r for r in results if r[1] is not None])
    print(f"\n总计: {passed}/{total} 通过")


if __name__ == "__main__":
    main()
