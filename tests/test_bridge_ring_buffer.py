from __future__ import annotations

import numpy as np

from app.infra.audio.bridge_ring_buffer import MockPcmRingBuffer, SharedMemoryPcmRingBuffer


def test_mock_pcm_ring_buffer_keeps_recent_frames_in_order() -> None:
    buffer = MockPcmRingBuffer(capacity_frames=4, channels=1)

    buffer.write(np.array([[1.0], [2.0]], dtype=np.float32))
    buffer.write(np.array([[3.0], [4.0], [5.0]], dtype=np.float32))

    assert buffer.snapshot().reshape(-1).tolist() == [2.0, 3.0, 4.0, 5.0]
    stats = buffer.stats()
    assert stats.buffered_frames == 4
    assert stats.total_written_frames == 5
    assert stats.dropped_frames == 1


def test_mock_pcm_ring_buffer_normalizes_channel_count() -> None:
    buffer = MockPcmRingBuffer(capacity_frames=4, channels=2)

    buffer.write(np.array([0.25, 0.5], dtype=np.float32))

    snapshot = buffer.snapshot()
    assert snapshot.shape == (2, 2)
    assert snapshot[:, 0].tolist() == [0.25, 0.5]
    assert snapshot[:, 1].tolist() == [0.25, 0.5]


def test_shared_memory_pcm_ring_buffer_can_be_attached_by_name() -> None:
    owner = SharedMemoryPcmRingBuffer.create(capacity_frames=4, channels=1)
    attached = SharedMemoryPcmRingBuffer.attach(
        name=owner.memory.name,
        capacity_frames=4,
        channels=1,
    )
    try:
        owner.write(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))

        assert attached.snapshot().reshape(-1).tolist() == [1.0, 2.0, 3.0]
        assert attached.stats().shared_memory_name == owner.memory.name
    finally:
        attached.close()
        owner.close_and_unlink()
