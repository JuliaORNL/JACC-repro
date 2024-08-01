
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx908
	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.section	.text._Z20test_function_kernelIdEvPT_iiiS0_S0_S0_,#alloc,#execinstr
	.protected	_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_ ; -- Begin function _Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
	.globl	_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
	.p2align	8
	.type	_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_,@function
_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_: ; @_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x3c
	s_load_dwordx4 s[12:15], s[4:5], 0x8
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s2, s0, 16
	s_and_b32 s0, s0, 0xffff
	s_and_b32 s1, s1, 0xffff
	s_mul_i32 s6, s6, s0
	s_mul_i32 s7, s7, s2
	v_add_u32_e32 v0, s6, v0
	v_add_u32_e32 v1, s7, v1
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v2, s8, v2
	v_cmp_gt_i32_e32 vcc, s12, v0
	v_cmp_gt_i32_e64 s[0:1], s13, v1
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cmp_gt_i32_e32 vcc, s14, v2
	s_and_b64 s[0:1], s[0:1], vcc
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x18
	v_cvt_f64_i32_e32 v[3:4], v1
	v_cvt_f64_i32_e32 v[5:6], v0
	v_cvt_f64_i32_e32 v[7:8], s13
	v_cvt_f64_i32_e32 v[11:12], s12
	s_waitcnt lgkmcnt(0)
	v_mul_f64 v[3:4], v[3:4], s[2:3]
	v_mul_f64 v[5:6], v[5:6], s[0:1]
	s_load_dwordx2 s[6:7], s[4:5], 0x28
	v_mul_f64 v[9:10], v[3:4], 0.5
	v_fma_f64 v[3:4], -v[7:8], s[2:3], v[3:4]
	v_cvt_f64_i32_e32 v[7:8], v2
	v_mul_f64 v[13:14], v[5:6], 0.5
	v_fma_f64 v[5:6], -v[11:12], s[0:1], v[5:6]
	v_mad_u64_u32 v[1:2], s[0:1], v2, s13, v[1:2]
	v_mad_u64_u32 v[0:1], s[0:1], v1, s12, v[0:1]
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	v_mul_f64 v[7:8], v[7:8], s[6:7]
	v_mul_f64 v[3:4], v[9:10], v[3:4]
	v_cvt_f64_i32_e32 v[9:10], s14
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	v_add_co_u32_e32 v0, vcc, s0, v0
	v_mul_f64 v[11:12], v[7:8], 0.5
	v_fma_f64 v[7:8], -v[9:10], s[6:7], v[7:8]
	v_fma_f64 v[3:4], v[13:14], v[5:6], v[3:4]
	v_fma_f64 v[2:3], v[11:12], v[7:8], v[3:4]
	v_mov_b32_e32 v4, s1
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB0_2:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 15
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z20test_function_kernelIdEvPT_iiiS0_S0_S0_,#alloc,#execinstr
.Lfunc_end0:
	.size	_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_, .Lfunc_end0-_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 356
; NumSgprs: 20
; NumVgprs: 15
; NumAgprs: 0
; TotalNumVgprs: 15
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 15
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.section	.text._Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,#alloc,#execinstr
	.protected	_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_ ; -- Begin function _Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.globl	_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.p2align	8
	.type	_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,@function
_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_: ; @_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
; %bb.0:
	s_load_dword s0, s[4:5], 0x4c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s0, 0xffff
	s_mul_i32 s6, s6, s1
	v_add_u32_e32 v0, s6, v0
	v_cmp_ne_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB1_4
; %bb.1:
	s_load_dword s1, s[4:5], 0x50
	s_load_dwordx4 s[12:15], s[4:5], 0x10
	s_lshr_b32 s0, s0, 16
	s_mul_i32 s7, s7, s0
	v_add_u32_e32 v1, s7, v1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s1, 0xffff
	s_add_i32 s0, s12, -1
	v_cmp_gt_i32_e32 vcc, s0, v0
	v_cmp_ne_u32_e64 s[0:1], 0, v1
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB1_4
; %bb.2:
	s_and_b32 s0, s3, 0xffff
	s_mul_i32 s8, s8, s0
	v_add_u32_e32 v2, s8, v2
	s_add_i32 s0, s13, -1
	v_cmp_gt_i32_e32 vcc, s0, v1
	v_cmp_ne_u32_e64 s[0:1], 0, v2
	s_add_i32 s2, s14, -1
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cmp_gt_i32_e32 vcc, s2, v2
	s_and_b64 s[0:1], s[0:1], vcc
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB1_4
; %bb.3:
	s_mul_i32 s6, s13, s12
	v_mul_lo_u32 v1, v1, s12
	v_mul_lo_u32 v2, v2, s6
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_ashr_i32 s13, s12, 31
	v_mov_b32_e32 v13, s13
	v_add3_u32 v4, v1, v0, v2
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[6:7], 3, v[4:5]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s3
	v_add_co_u32_e32 v8, vcc, s2, v6
	v_addc_co_u32_e32 v9, vcc, v0, v7, vcc
	global_load_dwordx2 v[10:11], v[8:9], off offset:8
	global_load_dwordx4 v[0:3], v[8:9], off offset:-8
	v_subrev_co_u32_e32 v12, vcc, s12, v4
	v_subb_co_u32_e32 v13, vcc, v5, v13, vcc
	v_lshlrev_b64 v[12:13], 3, v[12:13]
	v_mov_b32_e32 v14, s3
	v_add_co_u32_e32 v12, vcc, s2, v12
	v_addc_co_u32_e32 v13, vcc, v14, v13, vcc
	s_lshl_b64 s[8:9], s[12:13], 3
	v_mov_b32_e32 v15, s9
	v_add_co_u32_e32 v14, vcc, s8, v8
	v_addc_co_u32_e32 v15, vcc, v9, v15, vcc
	global_load_dwordx2 v[16:17], v[12:13], off
	global_load_dwordx2 v[18:19], v[14:15], off
	s_ashr_i32 s7, s6, 31
	v_mov_b32_e32 v12, s7
	v_subrev_co_u32_e32 v4, vcc, s6, v4
	v_subb_co_u32_e32 v5, vcc, v5, v12, vcc
	v_lshlrev_b64 v[4:5], 3, v[4:5]
	v_mov_b32_e32 v12, s3
	v_add_co_u32_e32 v4, vcc, s2, v4
	v_addc_co_u32_e32 v5, vcc, v12, v5, vcc
	s_lshl_b64 s[2:3], s[6:7], 3
	v_mov_b32_e32 v12, s3
	v_add_co_u32_e32 v8, vcc, s2, v8
	v_addc_co_u32_e32 v9, vcc, v9, v12, vcc
	global_load_dwordx2 v[12:13], v[4:5], off
	global_load_dwordx2 v[14:15], v[8:9], off
	s_load_dwordx8 s[4:11], s[4:5], 0x20
	s_waitcnt vmcnt(4)
	v_add_f64 v[0:1], v[0:1], v[10:11]
	s_waitcnt lgkmcnt(0)
	v_mul_f64 v[0:1], v[0:1], s[4:5]
	v_fma_f64 v[0:1], v[2:3], s[10:11], v[0:1]
	s_waitcnt vmcnt(2)
	v_add_f64 v[4:5], v[16:17], v[18:19]
	v_fma_f64 v[0:1], v[4:5], s[6:7], v[0:1]
	s_waitcnt vmcnt(0)
	v_add_f64 v[2:3], v[12:13], v[14:15]
	v_fma_f64 v[0:1], v[2:3], s[8:9], v[0:1]
	v_mov_b32_e32 v3, s1
	v_add_co_u32_e32 v2, vcc, s0, v6
	v_addc_co_u32_e32 v3, vcc, v3, v7, vcc
	global_store_dwordx2 v[2:3], v[0:1], off
.LBB1_4:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,#alloc,#execinstr
.Lfunc_end1:
	.size	_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_, .Lfunc_end1-_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 512
; NumSgprs: 20
; NumVgprs: 20
; NumAgprs: 0
; TotalNumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 20
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.section	.text._Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d,#alloc,#execinstr
	.protected	_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d ; -- Begin function _Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.globl	_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.p2align	8
	.type	_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d,@function
_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d: ; @_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
; %bb.0:
	s_load_dword s0, s[4:5], 0x4c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s0, 0xffff
	s_mul_i32 s6, s6, s1
	v_add_u32_e32 v0, s6, v0
	v_cmp_ne_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_6
; %bb.1:
	s_load_dword s1, s[4:5], 0x50
	s_load_dwordx4 s[12:15], s[4:5], 0x10
	s_lshr_b32 s0, s0, 16
	s_mul_i32 s7, s7, s0
	v_add_u32_e32 v1, s7, v1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s1, 0xffff
	s_add_i32 s0, s12, -1
	v_cmp_gt_i32_e32 vcc, s0, v0
	v_cmp_ne_u32_e64 s[0:1], 0, v1
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_6
; %bb.2:
	s_and_b32 s0, s3, 0xffff
	s_mul_i32 s8, s8, s0
	v_add_u32_e32 v2, s8, v2
	s_add_i32 s0, s13, -1
	v_cmp_gt_i32_e32 vcc, s0, v1
	v_cmp_ne_u32_e64 s[0:1], 0, v2
	s_add_i32 s2, s14, -1
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cmp_gt_i32_e32 vcc, s2, v2
	s_mov_b32 s6, 0
	s_and_b64 s[0:1], s[0:1], vcc
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_6
; %bb.3:
	v_mad_u64_u32 v[1:2], s[0:1], v2, s13, v[1:2]
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_mov_b32 s7, 0xc0080000
	v_mad_u64_u32 v[0:1], s[8:9], v1, s12, v[0:1]
	s_load_dwordx2 s[4:5], s[4:5], 0x38
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s3
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	v_add_co_u32_e32 v0, vcc, s2, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	global_load_dwordx2 v[0:1], v[0:1], off
	s_waitcnt vmcnt(0)
	v_add_f64 v[0:1], v[0:1], s[6:7]
	s_mov_b32 s7, 0x40080000
	v_and_b32_e32 v3, 0x7fffffff, v1
	v_mov_b32_e32 v2, v0
	v_div_scale_f64 v[4:5], s[2:3], s[6:7], s[6:7], v[2:3]
	v_div_scale_f64 v[2:3], vcc, v[2:3], s[6:7], v[2:3]
	v_rcp_f64_e32 v[6:7], v[4:5]
	v_fma_f64 v[8:9], -v[4:5], v[6:7], 1.0
	v_fma_f64 v[6:7], v[6:7], v[8:9], v[6:7]
	v_fma_f64 v[8:9], -v[4:5], v[6:7], 1.0
	v_fma_f64 v[6:7], v[6:7], v[8:9], v[6:7]
	v_mul_f64 v[8:9], v[2:3], v[6:7]
	v_fma_f64 v[2:3], -v[4:5], v[8:9], v[2:3]
	v_div_fmas_f64 v[2:3], v[2:3], v[6:7], v[8:9]
	v_div_fixup_f64 v[0:1], v[2:3], s[6:7], |v[0:1]|
	v_cmp_lt_f64_e32 vcc, s[4:5], v[0:1]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_6
; %bb.4:
	s_mov_b64 s[2:3], exec
	v_mbcnt_lo_u32_b32 v0, s2, 0
	v_mbcnt_hi_u32_b32 v0, s3, v0
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_b64 s[4:5], exec, vcc
	s_mov_b64 exec, s[4:5]
	s_cbranch_execz .LBB2_6
; %bb.5:
	s_bcnt1_i32_b64 s2, s[2:3]
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, s2
	global_atomic_add v0, v1, s[0:1]
.LBB2_6:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d,#alloc,#execinstr
.Lfunc_end2:
	.size	_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d, .Lfunc_end2-_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 484
; NumSgprs: 20
; NumVgprs: 10
; NumAgprs: 0
; TotalNumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 10
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.protected	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE ; @_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
	.type	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1yE ; @_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1yE
	.type	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1yE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1yE,#alloc
	.weak	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1yE
_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1yE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1yE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1yE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1yE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1yE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1yE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1yE
_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1yE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1yE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1yE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1yE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1yE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockDimE1yE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1yE
_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1yE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1yE, 1

	.protected	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1zE ; @_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1zE
	.type	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1zE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1zE,#alloc
	.weak	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1zE
_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1zE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1zE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1zE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1zE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1zE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1zE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1zE
_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1zE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1zE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1zE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1zE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1zE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockDimE1zE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1zE
_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1zE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1zE, 1

	.ident	"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.2 24012 af27734ed982b52a9f1be0f035ac91726fc697e4)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         12
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           8
        .value_kind:     by_value
      - .offset:         32
        .size:           8
        .value_kind:     by_value
      - .offset:         40
        .size:           8
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z20test_function_kernelIdEvPT_iiiS0_S0_S0_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     15
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           8
        .value_kind:     by_value
      - .offset:         40
        .size:           8
        .value_kind:     by_value
      - .offset:         48
        .size:           8
        .value_kind:     by_value
      - .offset:         56
        .size:           8
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           8
        .value_kind:     by_value
      - .offset:         40
        .size:           8
        .value_kind:     by_value
      - .offset:         48
        .size:           8
        .value_kind:     by_value
      - .offset:         56
        .size:           8
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx908

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.text
	.file	"laplacian.cpp"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI0_0:
	.long	1127219200                      # 0x43300000
	.long	1160773632                      # 0x45300000
	.long	0                               # 0x0
	.long	0                               # 0x0
.LCPI0_1:
	.quad	0x4330000000000000              # double 4503599627370496
	.quad	0x4530000000000000              # double 1.9342813113834067E+25
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI0_2:
	.quad	0x3ff0000000000000              # double 1
.LCPI0_3:
	.quad	0x3ec92a737110e454              # double 3.0000000000000001E-6
.LCPI0_4:
	.quad	0x412e848000000000              # double 1.0E+6
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_5:
	.long	0x447a0000                      # float 1000
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$120, %rsp
	.cfi_def_cfa_offset 176
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movb	$97, 15(%rsp)                   # 1-byte Folded Spill
	movl	$512, %eax                      # imm = 0x200
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movl	$256, %r13d                     # imm = 0x100
	movl	$1, %r14d
	cmpl	$2, %edi
	jl	.LBB0_3
# %bb.1:
	movq	%rsi, %r12
	movl	%edi, %ebp
	movq	8(%rsi), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	cltq
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	cmpl	$2, %ebp
	jne	.LBB0_4
# %bb.2:
	movl	$512, %r15d                     # imm = 0x200
	jmp	.LBB0_9
.LBB0_3:
	movl	$512, %r15d                     # imm = 0x200
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movl	$1, %ebp
	jmp	.LBB0_11
.LBB0_4:
	movq	16(%r12), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movslq	%eax, %r15
	cmpl	$4, %ebp
	jb	.LBB0_9
# %bb.5:
	movq	24(%r12), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	cltq
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpl	$4, %ebp
	je	.LBB0_9
# %bb.6:
	movq	32(%r12), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movq	%rax, %r13
	cmpl	$6, %ebp
	jb	.LBB0_9
# %bb.7:
	movq	40(%r12), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movl	%ebp, %ecx
	movq	%rax, %rbp
	movl	%ecx, 28(%rsp)                  # 4-byte Spill
	cmpl	$6, %ecx
	je	.LBB0_10
# %bb.67:
	movq	48(%r12), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
	callq	strtol
	movq	%rax, %r14
	cmpl	$8, 28(%rsp)                    # 4-byte Folded Reload
	jb	.LBB0_11
# %bb.68:
	movq	56(%r12), %rax
	movzbl	(%rax), %eax
	movb	%al, 15(%rsp)                   # 1-byte Spill
	jmp	.LBB0_11
.LBB0_9:
	movl	$1, %ebp
.LBB0_10:
	movl	$1, %r14d
.LBB0_11:
	movl	%r14d, %eax
	imull	%ebp, %eax
	imull	%r13d, %eax
	cmpl	$1025, %eax                     # imm = 0x401
	jl	.LBB0_17
# %bb.12:
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$33, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	%r13d, %esi
	callq	_ZNSolsEi
	movq	%rax, %r12
	movl	$.L.str.1, %esi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rdi
	movl	%ebp, %esi
	callq	_ZNSolsEi
	movq	%rax, %r12
	movl	$.L.str.1, %esi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rdi
	movl	%r14d, %esi
	callq	_ZNSolsEi
	movl	$_ZSt4cout, %edi
	movl	$.L.str.2, %esi
	movl	$15, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$1024, %esi                     # imm = 0x400
	callq	_ZNSolsEi
	movq	%rax, %r12
	movl	$.L.str.3, %esi
	movl	$15, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rdi
	movl	$1024, %esi                     # imm = 0x400
	callq	_ZNSolsEi
	movq	%rax, %r12
	movl	$.L.str.4, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	(%r12), %rax
	movq	-24(%rax), %rax
	movq	240(%r12,%rax), %r13
	testq	%r13, %r13
	je	.LBB0_76
# %bb.13:
	cmpb	$0, 56(%r13)
	je	.LBB0_15
# %bb.14:
	movzbl	67(%r13), %eax
	jmp	.LBB0_16
.LBB0_15:
	movq	%r13, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r13), %rax
	movq	%r13, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB0_16:
	movsbl	%al, %esi
	movq	%r12, %rdi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	movl	$1024, %r13d                    # imm = 0x400
	movl	$1, %r14d
	movl	$1, %ebp
.LBB0_17:
	cmpb	$97, 15(%rsp)                   # 1-byte Folded Reload
	jne	.LBB0_35
# %bb.18:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.5, %esi
	movl	$8, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$1, %esi
	callq	_ZNSolsEi
	movq	%rax, %rdi
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%rdi,%rax), %r12
	testq	%r12, %r12
	je	.LBB0_76
# %bb.19:
	cmpb	$0, 56(%r12)
	je	.LBB0_21
# %bb.20:
	movzbl	67(%r12), %eax
	jmp	.LBB0_22
.LBB0_21:
	movq	%rdi, %rbx
	movq	%r12, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movq	%r12, %rdi
	movl	$10, %esi
	callq	*48(%rax)
	movq	%rbx, %rdi
.LBB0_22:
	movsbl	%al, %esi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	movl	$_ZSt4cout, %edi
	movl	$.L.str.6, %esi
	movl	$17, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cout+240(%rax), %r12
	testq	%r12, %r12
	je	.LBB0_76
# %bb.23:
	cmpb	$0, 56(%r12)
	je	.LBB0_25
# %bb.24:
	movzbl	67(%r12), %eax
	jmp	.LBB0_26
.LBB0_25:
	movq	%r12, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movq	%r12, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB0_26:
	movsbl	%al, %esi
	movl	$_ZSt4cout, %edi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$11, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movq	32(%rsp), %rsi                  # 8-byte Reload
	callq	_ZNSo9_M_insertImEERSoT_
	movq	%rax, %r12
	movl	$.L.str.8, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rdi
	movq	%r15, %rsi
	callq	_ZNSo9_M_insertImEERSoT_
	movq	%rax, %r12
	movl	$.L.str.8, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rdi
	movq	40(%rsp), %rsi                  # 8-byte Reload
	callq	_ZNSo9_M_insertImEERSoT_
	movq	%rax, %rdi
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%rdi,%rax), %r12
	testq	%r12, %r12
	je	.LBB0_76
# %bb.27:
	cmpb	$0, 56(%r12)
	je	.LBB0_29
# %bb.28:
	movzbl	67(%r12), %eax
	jmp	.LBB0_30
.LBB0_29:
	movq	%rdi, %rbx
	movq	%r12, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movq	%r12, %rdi
	movl	$10, %esi
	callq	*48(%rax)
	movq	%rbx, %rdi
.LBB0_30:
	movsbl	%al, %esi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	movl	$_ZSt4cout, %edi
	movl	$.L.str.9, %esi
	movl	$14, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	%r13d, %esi
	callq	_ZNSolsEi
	movq	%rax, %r12
	movl	$.L.str.8, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rdi
	movl	%ebp, %esi
	callq	_ZNSolsEi
	movq	%rax, %r12
	movl	$.L.str.8, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rdi
	movl	%r14d, %esi
	callq	_ZNSolsEi
	movq	%rax, %rbx
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%rbx,%rax), %r12
	testq	%r12, %r12
	je	.LBB0_76
# %bb.31:
	cmpb	$0, 56(%r12)
	je	.LBB0_33
# %bb.32:
	movzbl	67(%r12), %eax
	jmp	.LBB0_34
.LBB0_33:
	movq	%r12, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movq	%r12, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB0_34:
	movsbl	%al, %esi
	movq	%rbx, %rdi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
.LBB0_35:
	movq	%r15, %r12
	imulq	32(%rsp), %r12                  # 8-byte Folded Reload
	imulq	40(%rsp), %r12                  # 8-byte Folded Reload
	shlq	$3, %r12
	leaq	56(%rsp), %rdi
	movq	%r12, %rsi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB0_77
# %bb.36:
	leaq	48(%rsp), %rdi
	movq	%r12, 80(%rsp)                  # 8-byte Spill
	movq	%r12, %rsi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB0_78
# %bb.37:
	movq	32(%rsp), %rbx                  # 8-byte Reload
	leaq	-1(%rbx), %rax
	movq	%rax, %xmm0
	movdqa	.LCPI0_0(%rip), %xmm6           # xmm6 = [1127219200,1160773632,0,0]
	punpckldq	%xmm6, %xmm0            # xmm0 = xmm0[0],xmm6[0],xmm0[1],xmm6[1]
	movapd	.LCPI0_1(%rip), %xmm2           # xmm2 = [4.503599627370496E+15,1.9342813113834067E+25]
	subpd	%xmm2, %xmm0
	movapd	%xmm0, %xmm3
	unpckhpd	%xmm0, %xmm3                    # xmm3 = xmm3[1],xmm0[1]
	addsd	%xmm0, %xmm3
	movsd	.LCPI0_2(%rip), %xmm0           # xmm0 = mem[0],zero
	movapd	%xmm0, %xmm5
	movapd	%xmm0, %xmm4
	divsd	%xmm3, %xmm5
	movsd	%xmm5, 96(%rsp)                 # 8-byte Spill
	leaq	-1(%r15), %rax
	movq	%rax, %xmm0
	punpckldq	%xmm6, %xmm0            # xmm0 = xmm0[0],xmm6[0],xmm0[1],xmm6[1]
	subpd	%xmm2, %xmm0
	movapd	%xmm0, %xmm3
	unpckhpd	%xmm0, %xmm3                    # xmm3 = xmm3[1],xmm0[1]
	addsd	%xmm0, %xmm3
	movapd	%xmm4, %xmm0
	divsd	%xmm3, %xmm0
	movapd	%xmm0, %xmm1
	movsd	%xmm0, 88(%rsp)                 # 8-byte Spill
	movq	40(%rsp), %r12                  # 8-byte Reload
	leaq	-1(%r12), %rax
	movq	%rax, %xmm0
	punpckldq	%xmm6, %xmm0            # xmm0 = xmm0[0],xmm6[0],xmm0[1],xmm6[1]
	subpd	%xmm2, %xmm0
	movapd	%xmm0, %xmm3
	unpckhpd	%xmm0, %xmm3                    # xmm3 = xmm3[1],xmm0[1]
	addsd	%xmm0, %xmm3
	movapd	%xmm4, %xmm2
	divsd	%xmm3, %xmm2
	movsd	%xmm2, 104(%rsp)                # 8-byte Spill
	movq	56(%rsp), %rdi
	movl	%ebx, %esi
	movl	%r15d, %edx
	movl	%r12d, %ecx
	movapd	%xmm5, %xmm0
	callq	_Z13test_functionIdEvPT_iiiS0_S0_S0_
	movq	48(%rsp), %rdi
	movq	56(%rsp), %rsi
	movl	%ebx, %edx
	movl	%r15d, %ecx
	movl	%r12d, %r8d
	movl	%r13d, %r9d
	movsd	96(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	88(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
	movsd	104(%rsp), %xmm2                # 8-byte Reload
                                        # xmm2 = mem[0],zero
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	48(%rsp), %rdi
	movsd	.LCPI0_3(%rip), %xmm3           # xmm3 = mem[0],zero
	movl	%ebx, %esi
	movl	%r15d, %edx
	movl	%r12d, %ecx
	movsd	96(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	88(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
	movsd	104(%rsp), %xmm2                # 8-byte Reload
                                        # xmm2 = mem[0],zero
	callq	_Z5checkIdEiPT_iiiS0_S0_S0_S0_
	testl	%eax, %eax
	je	.LBB0_43
# %bb.38:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.14, %esi
	movl	$53, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movsd	.LCPI0_3(%rip), %xmm0           # xmm0 = mem[0],zero
	movl	$_ZSt4cout, %edi
	callq	_ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rbx
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%rbx,%rax), %r12
	testq	%r12, %r12
	je	.LBB0_76
# %bb.39:
	cmpb	$0, 56(%r12)
	je	.LBB0_41
# %bb.40:
	movzbl	67(%r12), %eax
	jmp	.LBB0_42
.LBB0_41:
	movq	%r12, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movq	%r12, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB0_42:
	movsbl	%al, %esi
	movq	%rbx, %rdi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
.LBB0_43:
	leaq	64(%rsp), %rdi
	callq	hipEventCreate
	testl	%eax, %eax
	jne	.LBB0_79
# %bb.44:
	leaq	16(%rsp), %rdi
	callq	hipEventCreate
	testl	%eax, %eax
	movq	80(%rsp), %rsi                  # 8-byte Reload
	jne	.LBB0_80
# %bb.45:
	movq	32(%rsp), %rdx                  # 8-byte Reload
	leaq	-2(%rdx), %rax
	leaq	-2(%r15), %rcx
	imulq	%rax, %rcx
	leaq	(%rdx,%r15), %rax
	addq	$-4, %rax
	movq	40(%rsp), %rdx                  # 8-byte Reload
	leaq	-2(%rdx), %rdi
	imulq	%rcx, %rdi
	addq	%rdx, %rax
	addq	$-2, %rax
	shlq	$5, %rax
	subq	%rax, %rsi
	addq	$-64, %rsi
	movq	%rsi, 80(%rsp)                  # 8-byte Spill
	shlq	$3, %rdi
	movq	%rdi, 112(%rsp)                 # 8-byte Spill
	xorps	%xmm2, %xmm2
	movl	$1000, %ebx                     # imm = 0x3E8
	leaq	76(%rsp), %r12
	.p2align	4, 0x90
.LBB0_46:                               # =>This Inner Loop Header: Depth=1
	movss	%xmm2, 28(%rsp)                 # 4-byte Spill
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB0_69
# %bb.47:                               #   in Loop: Header=BB0_46 Depth=1
	movq	64(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	testl	%eax, %eax
	jne	.LBB0_70
# %bb.48:                               #   in Loop: Header=BB0_46 Depth=1
	movq	48(%rsp), %rdi
	movq	56(%rsp), %rsi
	movq	32(%rsp), %rdx                  # 8-byte Reload
                                        # kill: def $edx killed $edx killed $rdx
	movl	%r15d, %ecx
	movq	40(%rsp), %r8                   # 8-byte Reload
                                        # kill: def $r8d killed $r8d killed $r8
	movl	%r13d, %r9d
	movsd	96(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	88(%rsp), %xmm1                 # 8-byte Reload
                                        # xmm1 = mem[0],zero
	movsd	104(%rsp), %xmm2                # 8-byte Reload
                                        # xmm2 = mem[0],zero
	pushq	%r14
	.cfi_adjust_cfa_offset 8
	pushq	%rbp
	.cfi_adjust_cfa_offset 8
	callq	_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB0_71
# %bb.49:                               #   in Loop: Header=BB0_46 Depth=1
	movq	16(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	testl	%eax, %eax
	jne	.LBB0_72
# %bb.50:                               #   in Loop: Header=BB0_46 Depth=1
	movq	16(%rsp), %rdi
	callq	hipEventSynchronize
	testl	%eax, %eax
	jne	.LBB0_73
# %bb.51:                               #   in Loop: Header=BB0_46 Depth=1
	movq	64(%rsp), %rsi
	movq	16(%rsp), %rdx
	movq	%r12, %rdi
	callq	hipEventElapsedTime
	testl	%eax, %eax
	jne	.LBB0_74
# %bb.52:                               #   in Loop: Header=BB0_46 Depth=1
	movss	28(%rsp), %xmm2                 # 4-byte Reload
                                        # xmm2 = mem[0],zero,zero,zero
	addss	76(%rsp), %xmm2
	decl	%ebx
	jne	.LBB0_46
# %bb.53:
	movq	80(%rsp), %rcx                  # 8-byte Reload
	addq	112(%rsp), %rcx                 # 8-byte Folded Reload
	movzbl	15(%rsp), %eax                  # 1-byte Folded Reload
	cmpb	$97, %al
	jne	.LBB0_56
# %bb.54:
	movaps	%xmm2, %xmm0
	divss	.LCPI0_5(%rip), %xmm0
	cvtss2sd	%xmm0, %xmm0
	imulq	$1000, %rcx, %rax               # imm = 0x3E8
	testq	%rax, %rax
	js	.LBB0_59
# %bb.55:
	xorps	%xmm1, %xmm1
	cvtsi2ss	%rax, %xmm1
	jmp	.LBB0_60
.LBB0_56:
	cmpb	$98, %al
	jne	.LBB0_61
# %bb.57:
	imulq	$1000, %rcx, %rax               # imm = 0x3E8
	testq	%rax, %rax
	js	.LBB0_64
# %bb.58:
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	jmp	.LBB0_65
.LBB0_59:
	shrq	%rax
	xorps	%xmm1, %xmm1
	cvtsi2ss	%rax, %xmm1
	addss	%xmm1, %xmm1
.LBB0_60:
	divss	%xmm2, %xmm1
	cvtss2sd	%xmm1, %xmm1
	divsd	.LCPI0_4(%rip), %xmm1
	movl	$.L.str.15, %edi
	movb	$2, %al
	callq	printf
.LBB0_61:
	movq	48(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB0_66
.LBB0_62:
	movq	56(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB0_81
# %bb.63:
	xorl	%eax, %eax
	addq	$120, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_64:
	.cfi_def_cfa_offset 176
	shrq	%rax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	addss	%xmm0, %xmm0
.LBB0_65:
	divss	%xmm2, %xmm0
	cvtss2sd	%xmm0, %xmm0
	divsd	.LCPI0_4(%rip), %xmm0
	movl	$.L.str.16, %edi
	movb	$1, %al
	callq	printf
	movq	48(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	je	.LBB0_62
.LBB0_66:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	movq	48(%rsp), %rdi
	callq	hipFree
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$270, %esi                      # imm = 0x10E
	jmp	.LBB0_75
.LBB0_69:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	callq	hipDeviceSynchronize
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$248, %esi
	jmp	.LBB0_75
.LBB0_70:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	movq	64(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$249, %esi
	jmp	.LBB0_75
.LBB0_71:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	callq	hipGetLastError
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$251, %esi
	jmp	.LBB0_75
.LBB0_72:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	movq	16(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipEventRecord
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$252, %esi
	jmp	.LBB0_75
.LBB0_73:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	movq	16(%rsp), %rdi
	callq	hipEventSynchronize
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$253, %esi
	jmp	.LBB0_75
.LBB0_74:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	movq	64(%rsp), %rsi
	movq	16(%rsp), %rdx
	leaq	76(%rsp), %rdi
	callq	hipEventElapsedTime
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$254, %esi
.LBB0_75:
	callq	_ZNSolsEi
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movl	$-1, %edi
	callq	exit
.LBB0_76:
	callq	_ZSt16__throw_bad_castv
.LBB0_77:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	leaq	56(%rsp), %rdi
	movq	%r12, %rsi
	callq	hipMalloc
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$220, %esi
	jmp	.LBB0_75
.LBB0_78:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	leaq	48(%rsp), %rdi
	movq	80(%rsp), %rsi                  # 8-byte Reload
	callq	hipMalloc
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$221, %esi
	jmp	.LBB0_75
.LBB0_79:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	leaq	64(%rsp), %rdi
	callq	hipEventCreate
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$243, %esi
	jmp	.LBB0_75
.LBB0_80:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	leaq	16(%rsp), %rdi
	callq	hipEventCreate
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$244, %esi
	jmp	.LBB0_75
.LBB0_81:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	movq	56(%rsp), %rdi
	callq	hipFree
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$271, %esi                      # imm = 0x10F
	jmp	.LBB0_75
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.text._Z13test_functionIdEvPT_iiiS0_S0_S0_,"axG",@progbits,_Z13test_functionIdEvPT_iiiS0_S0_S0_,comdat
	.weak	_Z13test_functionIdEvPT_iiiS0_S0_S0_ # -- Begin function _Z13test_functionIdEvPT_iiiS0_S0_S0_
	.p2align	4, 0x90
	.type	_Z13test_functionIdEvPT_iiiS0_S0_S0_,@function
_Z13test_functionIdEvPT_iiiS0_S0_S0_:   # @_Z13test_functionIdEvPT_iiiS0_S0_S0_
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$184, %rsp
	.cfi_def_cfa_offset 224
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movsd	%xmm2, 40(%rsp)                 # 8-byte Spill
	movsd	%xmm1, 32(%rsp)                 # 8-byte Spill
	movsd	%xmm0, 24(%rsp)                 # 8-byte Spill
	movl	%ecx, %ebx
	movl	%edx, %r14d
	movl	%esi, %r15d
	movq	%rdi, %r12
	leal	-1(%r15), %eax
	shrl	$8, %eax
	incl	%eax
	movq	%r14, %rdi
	shlq	$32, %rdi
	orq	%rax, %rdi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	%ecx, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB1_2
# %bb.1:
	movq	%r12, 120(%rsp)
	movl	%r15d, 20(%rsp)
	movl	%r14d, 16(%rsp)
	movl	%ebx, 12(%rsp)
	movsd	24(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, 112(%rsp)
	movsd	32(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, 104(%rsp)
	movsd	40(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, 96(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	80(%rsp), %rdi
	leaq	64(%rsp), %rsi
	leaq	56(%rsp), %rdx
	leaq	48(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	80(%rsp), %rsi
	movl	88(%rsp), %edx
	movq	64(%rsp), %rcx
	movl	72(%rsp), %r8d
	leaq	128(%rsp), %r9
	movl	$_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_, %edi
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB1_2:
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB1_4
# %bb.3:
	addq	$184, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB1_4:
	.cfi_def_cfa_offset 224
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	callq	hipGetLastError
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$97, %esi
	callq	_ZNSolsEi
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movl	$-1, %edi
	callq	exit
.Lfunc_end1:
	.size	_Z13test_functionIdEvPT_iiiS0_S0_S0_, .Lfunc_end1-_Z13test_functionIdEvPT_iiiS0_S0_S0_
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function _Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_
.LCPI2_0:
	.quad	0x3ff0000000000000              # double 1
	.quad	0x3ff0000000000000              # double 1
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI2_1:
	.quad	0x3ff0000000000000              # double 1
.LCPI2_2:
	.quad	0xc000000000000000              # double -2
	.section	.text._Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_,"axG",@progbits,_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_,comdat
	.weak	_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_
	.p2align	4, 0x90
	.type	_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_,@function
_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_:  # @_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$240, %rsp
	.cfi_def_cfa_offset 288
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movapd	%xmm2, 208(%rsp)                # 16-byte Spill
	movsd	%xmm1, 24(%rsp)                 # 8-byte Spill
	movapd	%xmm0, 224(%rsp)                # 16-byte Spill
	movl	%r8d, %ebx
	movl	%ecx, %r14d
	movl	%edx, %r15d
	movq	%rsi, %r12
	movq	%rdi, %r13
	movl	288(%rsp), %r8d
	movl	296(%rsp), %ecx
	movl	%r9d, %r10d
	leal	-1(%r15), %eax
	xorl	%edx, %edx
	divl	%r9d
	movl	%eax, %esi
	leal	-1(%r14), %eax
	xorl	%edx, %edx
	divl	%r8d
	movl	%eax, %edi
	shlq	$32, %r8
	orq	%r10, %r8
	leal	1(%rsi), %r9d
	incl	%edi
	leal	-1(%rbx), %eax
	xorl	%edx, %edx
	divl	%ecx
                                        # kill: def $eax killed $eax def $rax
	leal	1(%rax), %esi
	shlq	$32, %rdi
	orq	%r9, %rdi
	movq	%r8, %rdx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB2_2
# %bb.1:
	movapd	224(%rsp), %xmm2                # 16-byte Reload
	unpcklpd	208(%rsp), %xmm2                # 16-byte Folded Reload
                                        # xmm2 = xmm2[0],mem[0]
	movapd	.LCPI2_0(%rip), %xmm0           # xmm0 = [1.0E+0,1.0E+0]
	divpd	%xmm2, %xmm0
	movsd	.LCPI2_1(%rip), %xmm1           # xmm1 = mem[0],zero
	movsd	24(%rsp), %xmm3                 # 8-byte Reload
                                        # xmm3 = mem[0],zero
	divsd	%xmm3, %xmm1
	divsd	%xmm3, %xmm1
	divpd	%xmm2, %xmm0
	movapd	%xmm0, %xmm2
	movlpd	%xmm0, 104(%rsp)
	movhpd	%xmm0, 88(%rsp)
	addsd	%xmm1, %xmm0
	unpckhpd	%xmm2, %xmm2                    # xmm2 = xmm2[1,1]
	addsd	%xmm0, %xmm2
	mulsd	.LCPI2_2(%rip), %xmm2
	movq	%r13, 120(%rsp)
	movq	%r12, 112(%rsp)
	movl	%r15d, 20(%rsp)
	movl	%r14d, 16(%rsp)
	movl	%ebx, 12(%rsp)
	movsd	%xmm1, 96(%rsp)
	movsd	%xmm2, 80(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	16(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	64(%rsp), %rdi
	leaq	48(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	32(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	64(%rsp), %rsi
	movl	72(%rsp), %edx
	movq	48(%rsp), %rcx
	movl	56(%rsp), %r8d
	leaq	128(%rsp), %r9
	movl	$_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_, %edi
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB2_2:
	addq	$240, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_, .Lfunc_end2-_Z9laplacianIdEvPT_S1_iiiiiiS0_S0_S0_
	.cfi_endproc
                                        # -- End function
	.section	.text._Z5checkIdEiPT_iiiS0_S0_S0_S0_,"axG",@progbits,_Z5checkIdEiPT_iiiS0_S0_S0_S0_,comdat
	.weak	_Z5checkIdEiPT_iiiS0_S0_S0_S0_  # -- Begin function _Z5checkIdEiPT_iiiS0_S0_S0_S0_
	.p2align	4, 0x90
	.type	_Z5checkIdEiPT_iiiS0_S0_S0_S0_,@function
_Z5checkIdEiPT_iiiS0_S0_S0_S0_:         # @_Z5checkIdEiPT_iiiS0_S0_S0_S0_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$240, %rsp
	.cfi_def_cfa_offset 288
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movsd	%xmm3, 56(%rsp)                 # 8-byte Spill
	movsd	%xmm2, 48(%rsp)                 # 8-byte Spill
	movsd	%xmm1, 40(%rsp)                 # 8-byte Spill
	movsd	%xmm0, 32(%rsp)                 # 8-byte Spill
	movl	%ecx, %ebx
	movl	%edx, %r14d
	movl	%esi, %r15d
	movq	%rdi, %r12
	leaq	8(%rsp), %rdi
	movl	$4, %esi
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB3_1
# %bb.2:
	movq	8(%rsp), %rdi
	movl	$4, %edx
	xorl	%esi, %esi
	callq	hipMemset
	testl	%eax, %eax
	jne	.LBB3_3
# %bb.4:
	leal	-1(%r15), %eax
	shrl	$8, %eax
	incl	%eax
	movq	%r14, %rdi
	shlq	$32, %rdi
	orq	%rax, %rdi
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	%ebx, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_6
# %bb.5:
	movq	8(%rsp), %rax
	movq	%rax, 152(%rsp)
	movq	%r12, 144(%rsp)
	movl	%r15d, 28(%rsp)
	movl	%r14d, 24(%rsp)
	movl	%ebx, 20(%rsp)
	movsd	32(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, 136(%rsp)
	movsd	40(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, 128(%rsp)
	movsd	48(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, 120(%rsp)
	movsd	56(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movsd	%xmm0, 112(%rsp)
	leaq	152(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	144(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	28(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	24(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	20(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	136(%rsp), %rax
	movq	%rax, 200(%rsp)
	leaq	128(%rsp), %rax
	movq	%rax, 208(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 216(%rsp)
	leaq	112(%rsp), %rax
	movq	%rax, 224(%rsp)
	leaq	96(%rsp), %rdi
	leaq	80(%rsp), %rsi
	leaq	72(%rsp), %rdx
	leaq	64(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	96(%rsp), %rsi
	movl	104(%rsp), %edx
	movq	80(%rsp), %rcx
	movl	88(%rsp), %r8d
	leaq	160(%rsp), %r9
	movl	$_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d, %edi
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	80(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB3_6:
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB3_7
# %bb.9:
	movl	$4, %edi
	callq	_Znam
	movq	%rax, %rbx
	movl	$1, (%rax)
	movq	8(%rsp), %rsi
	movl	$4, %edx
	movq	%rax, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	testl	%eax, %eax
	jne	.LBB3_10
# %bb.11:
	movl	(%rbx), %ebp
	movq	%rbx, %rdi
	callq	_ZdaPv
	movl	%ebp, %eax
	addq	$240, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB3_1:
	.cfi_def_cfa_offset 288
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	leaq	8(%rsp), %rdi
	callq	_ZL9hipMallocIiE10hipError_tPPT_m
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$136, %esi
	jmp	.LBB3_8
.LBB3_3:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	movq	8(%rsp), %rdi
	movl	$4, %edx
	xorl	%esi, %esi
	callq	hipMemset
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$137, %esi
	jmp	.LBB3_8
.LBB3_7:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rbx
	callq	hipGetLastError
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%rbx, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$139, %esi
	jmp	.LBB3_8
.LBB3_10:
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %r14
	movq	8(%rsp), %rsi
	movl	$4, %edx
	movq	%rbx, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
	movl	%eax, %edi
	callq	hipGetErrorString
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.11, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.12, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movl	$.L.str.13, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movl	$142, %esi
.LBB3_8:
	callq	_ZNSolsEi
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movl	$-1, %edi
	callq	exit
.Lfunc_end3:
	.size	_Z5checkIdEiPT_iiiS0_S0_S0_S0_, .Lfunc_end3-_Z5checkIdEiPT_iiiS0_S0_S0_S0_
	.cfi_endproc
                                        # -- End function
	.section	.text._Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_,"axG",@progbits,_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_,comdat
	.weak	_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_ # -- Begin function _Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_
	.p2align	4, 0x90
	.type	_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_,@function
_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_: # @_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_
	.cfi_startproc
# %bb.0:
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movl	%esi, 12(%rsp)
	movl	%edx, 8(%rsp)
	movl	%ecx, 4(%rsp)
	movsd	%xmm0, 80(%rsp)
	movsd	%xmm1, 72(%rsp)
	movsd	%xmm2, 64(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end4:
	.size	_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_, .Lfunc_end4-_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_
	.cfi_endproc
                                        # -- End function
	.section	.text._Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,"axG",@progbits,_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,comdat
	.weak	_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_ # -- Begin function _Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.p2align	4, 0x90
	.type	_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,@function
_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_: # @_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.cfi_startproc
# %bb.0:
	subq	$184, %rsp
	.cfi_def_cfa_offset 192
	movq	%rdi, 104(%rsp)
	movq	%rsi, 96(%rsp)
	movl	%edx, 12(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%r8d, 4(%rsp)
	movsd	%xmm0, 88(%rsp)
	movsd	%xmm1, 80(%rsp)
	movsd	%xmm2, 72(%rsp)
	movsd	%xmm3, 64(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	112(%rsp), %r9
	movl	$_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$200, %rsp
	.cfi_adjust_cfa_offset -200
	retq
.Lfunc_end5:
	.size	_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_, .Lfunc_end5-_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _ZL9hipMallocIiE10hipError_tPPT_m
	.type	_ZL9hipMallocIiE10hipError_tPPT_m,@function
_ZL9hipMallocIiE10hipError_tPPT_m:      # @_ZL9hipMallocIiE10hipError_tPPT_m
	.cfi_startproc
# %bb.0:
	movl	$4, %esi
	jmp	hipMalloc                       # TAILCALL
.Lfunc_end6:
	.size	_ZL9hipMallocIiE10hipError_tPPT_m, .Lfunc_end6-_ZL9hipMallocIiE10hipError_tPPT_m
	.cfi_endproc
                                        # -- End function
	.section	.text._Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d,"axG",@progbits,_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d,comdat
	.weak	_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d # -- Begin function _Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.p2align	4, 0x90
	.type	_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d,@function
_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d: # @_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.cfi_startproc
# %bb.0:
	subq	$184, %rsp
	.cfi_def_cfa_offset 192
	movq	%rdi, 104(%rsp)
	movq	%rsi, 96(%rsp)
	movl	%edx, 12(%rsp)
	movl	%ecx, 8(%rsp)
	movl	%r8d, 4(%rsp)
	movsd	%xmm0, 88(%rsp)
	movsd	%xmm1, 80(%rsp)
	movsd	%xmm2, 72(%rsp)
	movsd	%xmm3, 64(%rsp)
	leaq	104(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	96(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	112(%rsp), %r9
	movl	$_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$200, %rsp
	.cfi_adjust_cfa_offset -200
	retq
.Lfunc_end7:
	.size	_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d, .Lfunc_end7-_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_laplacian.cpp
	.type	_GLOBAL__sub_I_laplacian.cpp,@function
_GLOBAL__sub_I_laplacian.cpp:           # @_GLOBAL__sub_I_laplacian.cpp
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit                    # TAILCALL
.Lfunc_end8:
	.size	_GLOBAL__sub_I_laplacian.cpp, .Lfunc_end8-_GLOBAL__sub_I_laplacian.cpp
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB9_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle(%rip)
.LBB9_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end9:
	.size	__hip_module_ctor, .Lfunc_end9-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	__hip_gpubin_handle(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB10_2
# %bb.1:
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle(%rip)
.LBB10_2:
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end10:
	.size	__hip_module_dtor, .Lfunc_end10-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"WARNING: input thread block size "
	.size	.L.str, 34

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"*"
	.size	.L.str.1, 2

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	" exceeds limit "
	.size	.L.str.2, 16

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	", resetting to "
	.size	.L.str.3, 16

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"*1*1"
	.size	.L.str.4, 5

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"Kernel: "
	.size	.L.str.5, 9

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"Precision: double"
	.size	.L.str.6, 18

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"nx,ny,nz = "
	.size	.L.str.7, 12

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	", "
	.size	.L.str.8, 3

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"block sizes = "
	.size	.L.str.9, 15

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"HIP error: "
	.size	.L.str.10, 12

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	" in file "
	.size	.L.str.11, 10

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"laplacian.cpp"
	.size	.L.str.12, 14

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	":"
	.size	.L.str.13, 2

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"Correctness test failed. Pointwise error larger than "
	.size	.L.str.14, 54

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s \n"
	.size	.L.str.15, 68

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"%g"
	.size	.L.str.16, 3

	.type	_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_,@object # @_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
	.section	.rodata._Z20test_function_kernelIdEvPT_iiiS0_S0_S0_,"aG",@progbits,_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_,comdat
	.weak	_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
	.p2align	3, 0x0
_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_:
	.quad	_Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_
	.size	_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_, 8

	.type	_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,@object # @_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.section	.rodata._Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,"aG",@progbits,_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_,comdat
	.weak	_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.p2align	3, 0x0
_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_:
	.quad	_Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.size	_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_, 8

	.type	_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d,@object # @_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.section	.rodata._Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d,"aG",@progbits,_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d,comdat
	.weak	_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.p2align	3, 0x0
_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d:
	.quad	_Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.size	_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z20test_function_kernelIdEvPT_iiiS0_S0_S0_"
	.size	.L__unnamed_1, 44

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_"
	.size	.L__unnamed_2, 48

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d"
	.size	.L__unnamed_3, 40

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.hidden	__hip_gpubin_handle             # @__hip_gpubin_handle
	.type	__hip_gpubin_handle,@object
	.section	.bss.__hip_gpubin_handle,"aGw",@nobits,__hip_gpubin_handle,comdat
	.weak	__hip_gpubin_handle
	.p2align	3, 0x0
__hip_gpubin_handle:
	.quad	0
	.size	__hip_gpubin_handle, 8

	.section	.init_array,"aw",@init_array
	.p2align	3, 0x90
	.quad	_GLOBAL__sub_I_laplacian.cpp
	.quad	__hip_module_ctor
	.ident	"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.2 24012 af27734ed982b52a9f1be0f035ac91726fc697e4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z35__device_stub__test_function_kernelIdEvPT_iiiS0_S0_S0_
	.addrsig_sym _Z31__device_stub__laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.addrsig_sym _Z27__device_stub__check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.addrsig_sym _GLOBAL__sub_I_laplacian.cpp
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
	.addrsig_sym _ZSt4cerr
	.addrsig_sym _Z20test_function_kernelIdEvPT_iiiS0_S0_S0_
	.addrsig_sym _Z16laplacian_kernelIdEvPT_PKS0_iiiS0_S0_S0_S0_
	.addrsig_sym _Z12check_kernelIdEvPiPKT_iiiS1_S1_S1_d
	.addrsig_sym __hip_fatbin
	.addrsig_sym __hip_fatbin_wrapper

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
