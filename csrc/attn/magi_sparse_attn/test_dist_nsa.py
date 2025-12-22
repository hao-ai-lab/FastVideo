from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

# isort: split
from magi_attention.common import AttnRanges
from odin.common.parallel_state import ParallelState
from odin.config.parallel_config import ParallelConfig
from odin.testing import parameterize
from odin.testing.dist_common import DistTestBase, with_comms
from odin.testing.precision import assert_close

from dist_nsa import DistNSA, DistSparseAttnKey
from nsa_v2 import SparseAttnKey, native_sparse_attn_optim


class DistNSATestBase(DistTestBase):
    def init_pg(self):
        super().init_pg()

        assert (
            self.world_size == self.cp_inter_size * self.cp_intra_size
        ), f"Invalid world_size {self.world_size=} for {self.cp_inter_size=}x{self.cp_intra_size=}."

        device_meshes = {
            "default": init_device_mesh(
                device_type="cuda",
                mesh_shape=(self.world_size,),
                mesh_dim_names=("cp",),
            ),
            "hier-cp": init_device_mesh(
                device_type="cuda",
                mesh_shape=(self.cp_inter_size, self.cp_intra_size),
                mesh_dim_names=("cp_inter", "cp_intra"),
            ),
        }

        parallel_config = ParallelConfig(
            context_parallel_size=self.world_size,
            cp_high_bandwidth_domain_size=self.cp_intra_size,
        )

        self.parallel_state = ParallelState(
            device_meshes=device_meshes,
            config=parallel_config,
        )

        self.cp_inter_group = self.parallel_state._get_process_group("cp_inter")
        self.cp_intra_group = self.parallel_state._get_process_group("cp_intra")
        self.cp_inter_rank = self.parallel_state._get_local_rank("cp_inter")
        self.cp_intra_rank = self.parallel_state._get_local_rank("cp_intra")

    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def process_group(self):
        return dist.group.WORLD

    @property
    def world_size(self) -> int:
        return 8

    @property
    def cp_inter_size(self) -> int:
        return 4

    @property
    def cp_intra_size(self) -> int:
        return 2

    def _run_dist_nsa(
        self,
        test_case: dict[str, Any],
        mesh_idx: dict[str, int],
        save_tensors_flag: dict[str, bool],
        dtype: torch.dtype,
    ):
        # import pdb; pdb.set_trace()
        # extract arguments
        seqlen = test_case["seqlen"]
        num_heads = test_case["num_heads"]
        head_dim = test_case["head_dim"]
        q_ranges: AttnRanges = test_case["q_ranges"]
        k_ranges: AttnRanges = test_case["k_ranges"]
        block_ranges: AttnRanges = test_case["block_ranges"]
        topk = test_case["topk"]
        block_size = test_case["block_size"]
        use_cmp_attn = test_case["use_cmp_attn"]
        use_slt_attn = test_case["use_slt_attn"]
        use_win_attn = test_case["use_win_attn"]
        use_gating = test_case["use_gating"]

        # extract dist info
        cp_mesh = self.parallel_state[("cp_inter", "cp_intra")]
        cp_ring_mesh_idx, cp_ulysses_mesh_idx = mesh_idx["ring"], mesh_idx["ulysses"]
        cp_ring_size = (
            self.cp_inter_size if cp_ring_mesh_idx == 0 else self.cp_intra_size
        )
        cp_ulysses_size = (
            self.cp_inter_size if cp_ulysses_mesh_idx == 0 else self.cp_intra_size
        )
        cp_ring_rank = (
            self.cp_inter_rank if cp_ring_mesh_idx == 0 else self.cp_intra_rank
        )
        cp_ulysses_rank = (
            self.cp_inter_rank if cp_ulysses_mesh_idx == 0 else self.cp_intra_rank
        )
        cp_ring_group = (
            self.cp_inter_group if cp_ring_mesh_idx == 0 else self.cp_intra_group
        )
        cp_ulysses_group = (
            self.cp_inter_group if cp_ulysses_mesh_idx == 0 else self.cp_intra_group
        )

        # extract save tensors hook flags
        save_ring_ag_kv_for_win_attn = save_tensors_flag["save_ring_ag_kv_for_win_attn"]
        save_ring_ag_kv_for_cmp_attn = save_tensors_flag["save_ring_ag_kv_for_cmp_attn"]
        save_ring_ag_kv_for_slt_attn = save_tensors_flag["save_ring_ag_kv_for_slt_attn"]

        # construct dist nsa module
        dist_nsa = DistNSA()

        # compute pad size
        (
            cp_ring_pad_size,
            cp_ring_seqlen,
            cp_ring_shard_seqlen,
        ) = DistNSA.compute_pad_size_single(seqlen, cp_ring_size)
        (
            cp_ulysses_pad_size,
            cp_ulysses_seqlen,
            cp_ulysses_shard_seqlen,
        ) = DistNSA.compute_pad_size_single(cp_ring_shard_seqlen, cp_ulysses_size)

        # test compute pad size API
        cp_ring_pad_size_ref, cp_ulysses_pad_size_ref = DistNSA.compute_pad_size(
            seqlen,
            cp_mesh=cp_mesh,
            cp_ring_mesh_idx=cp_ring_mesh_idx,
            cp_ulysses_mesh_idx=cp_ulysses_mesh_idx,
        )
        assert cp_ring_pad_size == cp_ring_pad_size_ref
        assert cp_ulysses_pad_size == cp_ulysses_pad_size_ref

        # generate dist sparse attn key
        key = DistSparseAttnKey(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            block_ranges=block_ranges,
            block_size=block_size,
            block_topk=topk,
            use_cmp_attn=use_cmp_attn,
            use_slt_attn=use_slt_attn,
            use_win_attn=use_win_attn,
            use_gating=use_gating,
            parallel_state=self.parallel_state,
            cp_pad_size=cp_ulysses_pad_size,  # TODO: rename to cp_ulysses_pad_size
            gating_method="sigmoid",
            cp_ring_mesh_idx=cp_ring_mesh_idx,
            cp_ulysses_mesh_idx=cp_ulysses_mesh_idx,
            cp_ring_pad_size=cp_ring_pad_size,
            deterministic=True,  # NOTE: when testing, always use deterministic mode
            save_ring_ag_kv_for_win_attn=save_ring_ag_kv_for_win_attn,
            save_ring_ag_kv_for_cmp_attn=save_ring_ag_kv_for_cmp_attn,
            save_ring_ag_kv_for_slt_attn=save_ring_ag_kv_for_slt_attn,
        )

        assert key.cp_ring_rank == cp_ring_rank
        assert key.cp_ulysses_rank == cp_ulysses_rank
        assert key.cp_ring_size == cp_ring_size
        assert key.cp_ulysses_size == cp_ulysses_size

        print(
            f"[RANK {self.rank}]: "
            f"{cp_ring_rank=}, {cp_ulysses_rank=}, {cp_ring_size=}, {cp_ulysses_size=}, "
            f"{cp_ring_pad_size=}, {cp_ring_seqlen=}, {cp_ring_shard_seqlen=}, "
            f"{cp_ulysses_pad_size=}, {cp_ulysses_seqlen=}, {cp_ulysses_shard_seqlen=}, "
        )

        # construct global data
        global_q = torch.randn(
            (seqlen, num_heads, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        global_k = torch.randn(
            (seqlen, num_heads, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        global_v = torch.randn(
            (seqlen, num_heads, head_dim),
            dtype=dtype,
            device=self.device,
            requires_grad=True,
        )
        dist.all_reduce(global_q.data, op=dist.ReduceOp.AVG, group=self.process_group)
        dist.all_reduce(global_k.data, op=dist.ReduceOp.AVG, group=self.process_group)
        dist.all_reduce(global_v.data, op=dist.ReduceOp.AVG, group=self.process_group)

        # contruct global gating data
        if use_gating:
            if use_win_attn:
                global_g_win = torch.randn(
                    (seqlen, num_heads, 1),
                    dtype=dtype,
                    device=self.device,
                    requires_grad=True,
                )
                dist.all_reduce(
                    global_g_win.data, op=dist.ReduceOp.AVG, group=self.process_group
                )
            else:
                global_g_win = None
            if use_cmp_attn:
                global_g_cmp = torch.randn(
                    (seqlen, num_heads, 1),
                    dtype=dtype,
                    device=self.device,
                    requires_grad=True,
                )
                dist.all_reduce(
                    global_g_cmp.data, op=dist.ReduceOp.AVG, group=self.process_group
                )
            else:
                global_g_cmp = None
            if use_slt_attn:
                global_g_slt = torch.randn(
                    (seqlen, num_heads, 1),
                    dtype=dtype,
                    device=self.device,
                    requires_grad=True,
                )
                dist.all_reduce(
                    global_g_slt.data, op=dist.ReduceOp.AVG, group=self.process_group
                )
            else:
                global_g_slt = None
        else:
            global_g_win, global_g_cmp, global_g_slt = None, None, None

        # dispatch global data to construct local data
        # first, shard seqlen in ring mesh
        local_q, local_k, local_v, local_g_win, local_g_cmp, local_g_slt = [
            DistNSA.dispatch_single(
                x, group=cp_ring_group, cp_pad_size=cp_ring_pad_size
            )
            if x is not None
            else None
            for x in (
                global_q,
                global_k,
                global_v,
                global_g_win,
                global_g_cmp,
                global_g_slt,
            )
        ]
        # second, further shard seqlen in ulysses mesh
        local_q, local_k, local_v, local_g_win, local_g_cmp, local_g_slt = [
            DistNSA.dispatch_single(
                x, group=cp_ulysses_group, cp_pad_size=cp_ulysses_pad_size
            )
            if x is not None
            else None
            for x in (local_q, local_k, local_v, local_g_win, local_g_cmp, local_g_slt)
        ]

        # test dispatch API
        with torch.no_grad():
            local_q_ref = DistNSA.dispatch(
                x=global_q,
                cp_mesh=cp_mesh,
                cp_ring_pad_size=cp_ring_pad_size,
                cp_ulysses_pad_size=cp_ulysses_pad_size,
                cp_ring_mesh_idx=cp_ring_mesh_idx,
                cp_ulysses_mesh_idx=cp_ulysses_mesh_idx,
            )
            assert torch.equal(local_q, local_q_ref)

        # run dist nsa fwd to get local output
        local_o, *rest = dist_nsa(
            q=local_q,
            k=local_k,
            v=local_v,
            key=key,
            g_win=local_g_win,
            g_cmp=local_g_cmp,
            g_slt=local_g_slt,
        )

        # undispatch local output to get global output
        # first, undispatch in ulysses mesh
        global_o = DistNSA.undispatch_single(
            x=local_o, group=cp_ulysses_group, cp_pad_size=cp_ulysses_pad_size
        )
        # second, undispatch in ring mesh
        global_o = DistNSA.undispatch_single(
            x=global_o, group=cp_ring_group, cp_pad_size=cp_ring_pad_size
        )

        # test undispatch API
        with torch.no_grad():
            global_o_ref = DistNSA.undispatch(
                x=local_o,
                cp_mesh=cp_mesh,
                cp_ring_pad_size=cp_ring_pad_size,
                cp_ulysses_pad_size=cp_ulysses_pad_size,
                cp_ring_mesh_idx=cp_ring_mesh_idx,
                cp_ulysses_mesh_idx=cp_ulysses_mesh_idx,
            )
            assert torch.equal(global_o, global_o_ref)

        # run dist nsa bwd to get global grad
        global_do = torch.randn_like(global_o)
        dist.all_reduce(global_do.data, op=dist.ReduceOp.AVG, group=self.process_group)
        global_o.backward(global_do)
        global_dq, global_dk, global_dv = global_q.grad, global_k.grad, global_v.grad

        if use_gating:
            global_dg_win, global_dg_cmp, global_dg_slt = (
                global_g_win.grad if use_win_attn else None,
                global_g_cmp.grad if use_cmp_attn else None,
                global_g_slt.grad if use_slt_attn else None,
            )
        else:
            global_dg_win, global_dg_cmp, global_dg_slt = None, None, None

        # compare to ref
        self._compare_to_nsa_ref(
            key=key,
            global_q=global_q,
            global_k=global_k,
            global_v=global_v,
            global_g_win=global_g_win,
            global_g_cmp=global_g_cmp,
            global_g_slt=global_g_slt,
            global_o=global_o,
            global_do=global_do,
            global_dq=global_dq,
            global_dk=global_dk,
            global_dv=global_dv,
            global_dg_win=global_dg_win,
            global_dg_cmp=global_dg_cmp,
            global_dg_slt=global_dg_slt,
        )

    def _compare_to_nsa_ref(
        self,
        key: DistSparseAttnKey,
        global_q: torch.Tensor,
        global_k: torch.Tensor,
        global_v: torch.Tensor,
        global_g_win: torch.Tensor | None,
        global_g_cmp: torch.Tensor | None,
        global_g_slt: torch.Tensor | None,
        global_o: torch.Tensor,
        global_do: torch.Tensor,
        global_dq: torch.Tensor,
        global_dk: torch.Tensor,
        global_dv: torch.Tensor,
        global_dg_win: torch.Tensor | None,
        global_dg_cmp: torch.Tensor | None,
        global_dg_slt: torch.Tensor | None,
    ):
        # generate ref sparse attn key
        key_ref = SparseAttnKey(
            q_ranges=key.q_ranges,
            k_ranges=key.k_ranges,
            block_size=key.block_size,
            block_topk=key.block_topk,
            q_block_size=key.q_block_size,
            cmp_method=key.cmp_method,
            use_cmp_attn=key.use_cmp_attn,
            use_slt_attn=key.use_slt_attn,
            use_win_attn=key.use_win_attn,
            use_nsa=key.use_nsa,
            use_gating=key.use_gating,
            parallel_state=self.parallel_state,
            cp_pad_size=0,
            gating_method=key.gating_method,
            block_ranges=key.block_ranges,
            deterministic=key.deterministic,
        )

        # clean global grad
        global_q.grad, global_k.grad, global_v.grad = None, None, None
        if key_ref.use_gating:
            if key_ref.use_win_attn:
                global_g_win.grad = None  # type: ignore[union-attr]
            if key_ref.use_cmp_attn:
                global_g_cmp.grad = None  # type: ignore[union-attr]
            if key_ref.use_slt_attn:
                global_g_slt.grad = None  # type: ignore[union-attr]

        # run ref nsa fwd
        global_o_ref = native_sparse_attn_optim(
            q=global_q,
            k=global_k,
            v=global_v,
            key=key_ref,
            g_win=global_g_win,
            g_cmp=global_g_cmp,
            g_slt=global_g_slt,
        )

        # run ref nsa bwd
        global_o_ref.backward(global_do)
        global_dq_ref, global_dk_ref, global_dv_ref = (
            global_q.grad,
            global_k.grad,
            global_v.grad,
        )
        # FIXME: why different ranks result in different ref grads ?
        dist.all_reduce(
            global_dq_ref.data,  # type: ignore[attr-defined]
            op=dist.ReduceOp.AVG,
            group=self.process_group,
        )
        dist.all_reduce(
            global_dk_ref.data,  # type: ignore[attr-defined]
            op=dist.ReduceOp.AVG,
            group=self.process_group,
        )
        dist.all_reduce(
            global_dv_ref.data,  # type: ignore[attr-defined]
            op=dist.ReduceOp.AVG,
            group=self.process_group,
        )
        if key_ref.use_gating:
            global_dg_win_ref, global_dg_cmp_ref, global_dg_slt_ref = (
                global_g_win.grad if key_ref.use_win_attn else None,  # type: ignore[union-attr]
                global_g_cmp.grad if key_ref.use_cmp_attn else None,  # type: ignore[union-attr]
                global_g_slt.grad if key_ref.use_slt_attn else None,  # type: ignore[union-attr]
            )
        else:
            global_dg_win_ref, global_dg_cmp_ref, global_dg_slt_ref = None, None, None

        # assert close
        err_msg_list = []
        try:
            assert_close(
                global_o,
                global_o_ref,
                atol=1e-5,
                rtol=5e-3,
                mismatch_threshold=0.05,
                test_case="global_o",
            )
        except Exception as e:
            err_msg_list.append(str(e))
        try:
            assert_close(
                global_dq,
                global_dq_ref,
                atol=1e-5,
                rtol=5e-2,
                mismatch_threshold=0.1,
                test_case="global_dq",
            )
        except Exception as e:
            err_msg_list.append(str(e))
        try:
            assert_close(
                global_dk,
                global_dk_ref,
                atol=1e-5,
                rtol=5e-2,
                mismatch_threshold=0.1,
                test_case="global_dk",
            )
        except Exception as e:
            err_msg_list.append(str(e))
        try:
            assert_close(
                global_dv,
                global_dv_ref,
                atol=1e-5,
                rtol=5e-2,
                mismatch_threshold=0.1,
                test_case="global_dv",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        if key_ref.use_gating:
            try:
                assert_close(
                    global_dg_win,
                    global_dg_win_ref,
                    atol=1e-5,
                    rtol=5e-3,
                    mismatch_threshold=0.05,
                    test_case="global_dg_win",
                )
            except Exception as e:
                err_msg_list.append(str(e))
            try:
                assert_close(
                    global_dg_cmp,
                    global_dg_cmp_ref,
                    atol=1e-5,
                    rtol=5e-3,
                    mismatch_threshold=0.05,
                    test_case="global_dg_cmp",
                )
            except Exception as e:
                err_msg_list.append(str(e))
            try:
                assert_close(
                    global_dg_slt,
                    global_dg_slt_ref,
                    atol=1e-5,
                    rtol=5e-3,
                    mismatch_threshold=0.05,
                    test_case="global_dg_slt",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        if err_msg_list:
            raise AssertionError("\n".join(err_msg_list))


class TestDistNSAWithWorldSize8(DistNSATestBase):
    @property
    def world_size(self) -> int:
        return 8
    
    @property
    def cp_inter_size(self) -> int:
        return 4

    @property
    def cp_intra_size(self) -> int:
        return 2

    # 在这里贴上参数化配置和 8卡的 skip 限制
    @skip_if_lt_x_gpu(8)
    @with_comms
    @parameterize(
        "test_case",
        [
            {
                "name": "case1",
                "seqlen": 511,
                "num_heads": 12,
                "head_dim": 128,
                "q_ranges": AttnRanges.from_ranges([[0, 511]]),
                "k_ranges": AttnRanges.from_ranges([[0, 511]]),
                "block_ranges": AttnRanges.from_ranges([[0, 32], [32, 64], [64, 128], [128, 160], [160, 192], [192, 256], [256, 288], [288, 384], [384, 448], [448, 511]]),
                "window_size": 128,
                "topk": 8,
                "block_size": 128,
                "use_cmp_attn": True,
                "use_win_attn": True,
                "use_slt_attn": True,
                "use_gating": True,
            },
        ],
    )
    @parameterize("mesh_idx", [{"ring": 0, "ulysses": 1}])
    @parameterize("save_tensors_flag", [{"save_ring_ag_kv_for_win_attn": False, "save_ring_ag_kv_for_cmp_attn": True, "save_ring_ag_kv_for_slt_attn": False}])
    @parameterize("dtype", [torch.bfloat16])
    def test_dist_nsa(self, *args, **kwargs):
        # 调用基类的纯逻辑方法
        self._run_dist_nsa(*args, **kwargs)


class TestDistNSAWithWorldSize6(TestDistNSAWithWorldSize8):
    @property
    def world_size(self) -> int:
        return 6

    @property
    def cp_inter_size(self) -> int:
        return 2

    @property
    def cp_intra_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(6)
    def test_dist_nsa(self, *args, **kwargs):
        super().test_dist_nsa(*args, **kwargs)


class TestDistNSAWithWorldSize4(TestDistNSAWithWorldSize8):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def cp_inter_size(self) -> int:
        return 2

    @property
    def cp_intra_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(4)
    def test_dist_nsa(self, *args, **kwargs):
        super().test_dist_nsa(*args, **kwargs)


class TestDistNSAWithWorldSize3(TestDistNSAWithWorldSize8):
    @property
    def world_size(self) -> int:
        return 3

    @property
    def cp_inter_size(self) -> int:
        return 1

    @property
    def cp_intra_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(3)
    def test_dist_nsa(self, *args, **kwargs):
        super().test_dist_nsa(*args, **kwargs)


class TestDistNSAWithWorldSize2(TestDistNSAWithWorldSize8):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def cp_inter_size(self) -> int:
        return 2

    @property
    def cp_intra_size(self) -> int:
        return 1

    @skip_if_lt_x_gpu(2)
    def test_dist_nsa(self, *args, **kwargs):
        super().test_dist_nsa(*args, **kwargs)


class TestDistNSAWithWorldSize1(DistNSATestBase):
    file_name = "tmp"
    @property
    def world_size(self) -> int:
        return 1

    @property
    def cp_inter_size(self) -> int:
        return 1

    @property
    def cp_intra_size(self) -> int:
        return 1

    @skip_if_lt_x_gpu(1) 
    @with_comms
    @parameterize(
        "test_case",
        [
            {
                "name": "case1",
                "seqlen": 511, 
                # ... 同样的参数 ...
                "num_heads": 12, "head_dim": 128,
                "q_ranges": AttnRanges.from_ranges([[0, 511]]),
                "k_ranges": AttnRanges.from_ranges([[0, 511]]),
                "block_ranges": AttnRanges.from_ranges([[0, 32], [32, 64], [64, 128], [128, 160], [160, 192], [192, 256], [256, 288], [288, 384], [384, 448], [448, 511]]),
                "window_size": 128, "topk": 8, "block_size": 128,
                "use_cmp_attn": True, "use_win_attn": True, "use_slt_attn": True, "use_gating": True,
            },
        ],
    )
    @parameterize("mesh_idx", [{"ring": 0, "ulysses": 0}]) # 注意：World Size 1 时 ring/ulysses 索引只能是0
    @parameterize("save_tensors_flag", [{"save_ring_ag_kv_for_win_attn": False, "save_ring_ag_kv_for_cmp_attn": True, "save_ring_ag_kv_for_slt_attn": False}])
    @parameterize("dtype", [torch.bfloat16])
    def test_dist_nsa(self, *args, **kwargs):
        # 还是调用那个纯逻辑方法
        self._run_dist_nsa(*args, **kwargs)


if __name__ == "__main__":
    run_tests()


# if __name__ == "__main__":
#     import os
    
#     # 1. 手动设置分布式环境变量 (模拟 World Size = 1)
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"
#     os.environ["RANK"] = "0"
#     os.environ["WORLD_SIZE"] = "1"
#     os.environ["LOCAL_RANK"] = "0"

#     # 2. 手动初始化进程组 (绕过 DistTestBase 的复杂逻辑)
#     if not dist.is_initialized():
#         dist.init_process_group(backend="nccl", init_method="env://")
    
#     torch.cuda.set_device(0)

#     # 3. 实例化你的测试类
#     # 注意：这里我们不调用 run_test，而是直接把类当普通对象用
#     tester = TestDistNSAWithWorldSize1()
    
#     # 手动补丁：有些 DistTestBase 依赖 self.rank 属性，而不仅仅是 dist.get_rank()
#     tester.rank = 0 
    
#     print(">>>手动初始化环境完成，开始运行 init_pg...")
    
#     # 4. 手动调用 init_pg 来建立 Mesh 和 ParallelState
#     tester.init_pg()

#     print(">>> 环境建立完毕，准备运行测试逻辑 (pdb 应该可以工作了)...")

#     # 5. 手动构造参数 (把 case1 的内容硬编码在这里，或者直接复制下来)
#     # 这是为了单独调试这一个 case
#     test_case_args = {
#         "name": "case1",
#         "seqlen": 511,
#         "num_heads": 12,
#         "head_dim": 128,
#         "q_ranges": AttnRanges.from_ranges([[0, 511]]),
#         "k_ranges": AttnRanges.from_ranges([[0, 511]]),
#         "block_ranges": AttnRanges.from_ranges(
#             [[0, 32], [32, 64], [64, 128], [128, 160], [160, 192], 
#              [192, 256], [256, 288], [288, 384], [384, 448], [448, 511]]
#         ),
#         "window_size": 128,
#         "topk": 8,
#         "block_size": 128,
#         "use_cmp_attn": True,
#         "use_win_attn": False,
#         "use_slt_attn": True,
#         "use_gating": True,
#     }
    
#     mesh_idx_args = {"ring": 0, "ulysses": 0} # 1卡只能是0
    
#     save_tensors_flag_args = {
#         "save_ring_ag_kv_for_win_attn": False,
#         "save_ring_ag_kv_for_cmp_attn": True,
#         "save_ring_ag_kv_for_slt_attn": False,
#     }

#     # 6. 直接调用你在上一步抽离出来的逻辑函数
#     # 此时代码在主进程运行，pdb.set_trace() 会正常拦截键盘输入
#     tester._run_dist_nsa(
#         test_case=test_case_args,
#         mesh_idx=mesh_idx_args,
#         save_tensors_flag=save_tensors_flag_args,
#         dtype=torch.bfloat16
#     )

#     print(">>> 测试结束")
    
#     # 清理
#     dist.destroy_process_group()
