Using AMDGPU as back end
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:32 within `parallel_for`
define void @julia_parallel_for_3845([3 x i64]* nocapture noundef nonnull readonly align 8 dereferenceable(24) %0, {}* noundef nonnull align 8 dereferenceable(40) %1, {}* noundef nonnull align 8 dereferenceable(40) %2, {}* noundef nonnull align 8 dereferenceable(40) %3, {}* noundef nonnull align 8 dereferenceable(40) %4, {}* noundef nonnull align 8 dereferenceable(24) %5, double %6, double %7, double %8, double %9, double %10, double %11) #0 {
top:
  %12 = alloca [2 x {}*], align 8
  %gcframe176 = alloca [16 x {}*], align 16
  %gcframe176.sub = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 0
  %.sub = getelementptr inbounds [2 x {}*], [2 x {}*]* %12, i64 0, i64 0
  %13 = bitcast [16 x {}*]* %gcframe176 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %13, i8 0, i64 128, i1 true)
  %14 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 14
  %15 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 13
  %16 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 2
  %17 = bitcast {}** %16 to [2 x {}*]*
  %18 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %19 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %20 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %21 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %22 = alloca { [1 x i64], i8 addrspace(1)*, i64 }, align 8
  %23 = alloca { { i64, {}*, {}* } }, align 8
  %24 = alloca { [2 x [3 x i64]], [2 x {}*] }, align 8
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #16
  %tls_ppgcstack = getelementptr i8, i8* %thread_ptr, i64 -8
  %25 = bitcast i8* %tls_ppgcstack to {}****
  %tls_pgcstack = load {}***, {}**** %25, align 8
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:35 within `parallel_for`
; ┌ @ tuple.jl:92 within `indexed_iterate` @ tuple.jl:92
   %26 = bitcast [16 x {}*]* %gcframe176 to i64*
   store i64 56, i64* %26, align 16
   %27 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 1
   %28 = bitcast {}** %27 to {}***
   %29 = load {}**, {}*** %tls_pgcstack, align 8
   store {}** %29, {}*** %28, align 8
   %30 = bitcast {}*** %tls_pgcstack to {}***
   store {}** %gcframe176.sub, {}*** %30, align 8
   %31 = getelementptr inbounds [3 x i64], [3 x i64]* %0, i64 0, i64 0
; │ @ tuple.jl:92 within `indexed_iterate`
   %32 = getelementptr inbounds [3 x i64], [3 x i64]* %0, i64 0, i64 1
   %33 = getelementptr inbounds [3 x i64], [3 x i64]* %0, i64 0, i64 2
; └
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:36 within `parallel_for`
; ┌ @ promotion.jl:533 within `min`
; │┌ @ int.jl:83 within `<`
    %unbox = load i64, i64* %31, align 8
; │└
; │┌ @ essentials.jl:647 within `ifelse`
    %34 = call i64 @llvm.smin.i64(i64 %unbox, i64 32)
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:37 within `parallel_for`
; ┌ @ promotion.jl:533 within `min`
; │┌ @ int.jl:83 within `<`
    %unbox2 = load i64, i64* %32, align 8
; │└
; │┌ @ essentials.jl:647 within `ifelse`
    %35 = call i64 @llvm.smin.i64(i64 %unbox2, i64 32)
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:39 within `parallel_for`
; ┌ @ int.jl:97 within `/`
; │┌ @ float.jl:294 within `float`
; ││┌ @ float.jl:268 within `AbstractFloat`
; │││┌ @ float.jl:159 within `Float64`
      %36 = sitofp i64 %unbox to double
      %37 = sitofp i64 %34 to double
; │└└└
; │ @ int.jl:97 within `/` @ float.jl:412
   %38 = fdiv double %36, %37
; └
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:395 within `round`
    %39 = call double @llvm.ceil.f64(double %38)
; │└
; │┌ @ float.jl:902 within `trunc`
; ││┌ @ float.jl:537 within `<=`
     %40 = fcmp ult double %39, 0xC3E0000000000000
; ││└
    %41 = fcmp uge double %39, 0x43E0000000000000
    %42 = or i1 %40, %41
    br i1 %42, label %L21, label %L19

L19:                                              ; preds = %top
; ││ @ float.jl:903 within `trunc`
; ││┌ @ float.jl:336 within `unsafe_trunc`
     %43 = fptosi double %39 to i64
     %44 = freeze i64 %43
; └└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:40 within `parallel_for`
; ┌ @ int.jl:97 within `/`
; │┌ @ float.jl:294 within `float`
; ││┌ @ float.jl:268 within `AbstractFloat`
; │││┌ @ float.jl:159 within `Float64`
      %45 = sitofp i64 %unbox2 to double
      %46 = sitofp i64 %35 to double
; │└└└
; │ @ int.jl:97 within `/` @ float.jl:412
   %47 = fdiv double %45, %46
; └
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:395 within `round`
    %48 = call double @llvm.ceil.f64(double %47)
; │└
; │┌ @ float.jl:902 within `trunc`
; ││┌ @ float.jl:537 within `<=`
     %49 = fcmp ult double %48, 0xC3E0000000000000
; ││└
    %50 = fcmp uge double %48, 0x43E0000000000000
    %51 = or i1 %49, %50
    br i1 %51, label %L38, label %L36

L21:                                              ; preds = %top
    %52 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 7
    %53 = bitcast {}** %52 to [3 x {}*]*
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:39 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:905 within `trunc`
    %ptls_field187 = getelementptr inbounds {}**, {}*** %tls_pgcstack, i64 2
    %54 = bitcast {}*** %ptls_field187 to i8**
    %ptls_load188189 = load i8*, i8** %54, align 8
    %box123 = call noalias nonnull dereferenceable(16) {}* @ijl_gc_pool_alloc(i8* %ptls_load188189, i32 752, i32 16) #14
    %55 = bitcast {}* %box123 to i64*
    %56 = getelementptr inbounds i64, i64* %55, i64 -1
    store atomic i64 139635484756704, i64* %56 unordered, align 8
    %57 = bitcast {}* %box123 to double*
    store double %39, double* %57, align 8
    %58 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %box123, {}** %58, align 8
    call void @j_InexactError_3854([3 x {}*]* noalias nocapture noundef nonnull sret([3 x {}*]) %53, {}* inttoptr (i64 139635639651088 to {}*), {}* readonly inttoptr (i64 139635484757504 to {}*), {}* nonnull readonly %box123)
    %ptls_load162190191 = load i8*, i8** %54, align 8
    %box125 = call noalias nonnull dereferenceable(32) {}* @ijl_gc_pool_alloc(i8* %ptls_load162190191, i32 800, i32 32) #14
    %59 = bitcast {}* %box125 to i64*
    %60 = getelementptr inbounds i64, i64* %59, i64 -1
    store atomic i64 139635415075680, i64* %60 unordered, align 8
    %61 = bitcast {}* %box125 to i8*
    %62 = bitcast {}** %52 to i8*
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %61, i8* noundef nonnull align 8 dereferenceable(24) %62, i64 24, i1 false)
    call void @ijl_throw({}* %box125)
    unreachable

L36:                                              ; preds = %L19
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:40 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:903 within `trunc`
; ││┌ @ float.jl:336 within `unsafe_trunc`
     %63 = fptosi double %48 to i64
     %64 = freeze i64 %63
; └└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:41 within `parallel_for`
; ┌ @ int.jl:97 within `/`
; │┌ @ float.jl:294 within `float`
; ││┌ @ float.jl:268 within `AbstractFloat`
; │││┌ @ float.jl:159 within `Float64`
      %unbox8 = load i64, i64* %33, align 8
      %65 = sitofp i64 %unbox8 to double
; └└└└
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:902 within `trunc`
; ││┌ @ float.jl:537 within `<=`
     %66 = fcmp ult double %65, 0xC3E0000000000000
; ││└
    %67 = fcmp uge double %65, 0x43E0000000000000
    %68 = or i1 %66, %67
    br i1 %68, label %L54, label %L52

L38:                                              ; preds = %L19
    %69 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 10
    %70 = bitcast {}** %69 to [3 x {}*]*
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:40 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:905 within `trunc`
    %ptls_field164182 = getelementptr inbounds {}**, {}*** %tls_pgcstack, i64 2
    %71 = bitcast {}*** %ptls_field164182 to i8**
    %ptls_load165183184 = load i8*, i8** %71, align 8
    %box117 = call noalias nonnull dereferenceable(16) {}* @ijl_gc_pool_alloc(i8* %ptls_load165183184, i32 752, i32 16) #14
    %72 = bitcast {}* %box117 to i64*
    %73 = getelementptr inbounds i64, i64* %72, i64 -1
    store atomic i64 139635484756704, i64* %73 unordered, align 8
    %74 = bitcast {}* %box117 to double*
    store double %48, double* %74, align 8
    %75 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %box117, {}** %75, align 8
    call void @j_InexactError_3854([3 x {}*]* noalias nocapture noundef nonnull sret([3 x {}*]) %70, {}* inttoptr (i64 139635639651088 to {}*), {}* readonly inttoptr (i64 139635484757504 to {}*), {}* nonnull readonly %box117)
    %ptls_load168185186 = load i8*, i8** %71, align 8
    %box119 = call noalias nonnull dereferenceable(32) {}* @ijl_gc_pool_alloc(i8* %ptls_load168185186, i32 800, i32 32) #14
    %76 = bitcast {}* %box119 to i64*
    %77 = getelementptr inbounds i64, i64* %76, i64 -1
    store atomic i64 139635415075680, i64* %77 unordered, align 8
    %78 = bitcast {}* %box119 to i8*
    %79 = bitcast {}** %69 to i8*
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %78, i8* noundef nonnull align 16 dereferenceable(24) %79, i64 24, i1 false)
    call void @ijl_throw({}* %box119)
    unreachable

L52:                                              ; preds = %L36
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:41 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:903 within `trunc`
; ││┌ @ float.jl:336 within `unsafe_trunc`
     %80 = fptosi double %65 to i64
     %81 = freeze i64 %80
; └└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:170 within `macro expansion`
; │┌ @ tuple.jl:294 within `map` @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3847({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %18, {}* nonnull %1)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3847({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %19, {}* nonnull %2)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3847({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %20, {}* nonnull %3)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3847({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %21, {}* nonnull %4)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3848({ [1 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [1 x i64], i8 addrspace(1)*, i64 }) %22, {}* nonnull %5)
; │└└└└└
; │ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:172 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/compiler/codegen.jl:132 within `hipfunction`
    call void @"j_#hipfunction#37_3849"({ { i64, {}*, {}* } }* noalias nocapture noundef nonnull sret({ { i64, {}*, {}* } }) %23, [2 x {}*]* noalias nocapture noundef nonnull %17)
; │└
; │ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream`
     %82 = call nonnull {}* @"j_task_local_state!_3850"()
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:62
; │││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
      %83 = bitcast {}* %82 to { i64, i32, {}*, i32 }*
      %84 = getelementptr inbounds { i64, i32, {}*, i32 }, { i64, i32, {}*, i32 }* %83, i64 0, i32 1
      %85 = load i32, i32* %84, align 4
; │││└
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; │││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
      %86 = bitcast {}* %82 to i8*
      %getfield_addr30 = getelementptr inbounds i8, i8* %86, i64 40
      %87 = bitcast i8* %getfield_addr30 to {}**
      %getfield31 = load atomic {}*, {}** %87 unordered, align 8
; │││└
; │││┌ @ abstractarray.jl:1294 within `getindex`
; ││││┌ @ indices.jl:350 within `to_indices` @ indices.jl:354
; │││││┌ @ indices.jl:359 within `_to_indices1`
; ││││││┌ @ indices.jl:277 within `to_index` @ indices.jl:292
; │││││││┌ @ number.jl:7 within `convert`
; ││││││││┌ @ boot.jl:784 within `Int64`
; │││││││││┌ @ boot.jl:703 within `toInt64`
            %88 = sext i32 %85 to i64
; ││││└└└└└└
; ││││ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
      %89 = add nsw i64 %88, -1
      %90 = bitcast {}* %getfield31 to { i8*, i64, i16, i16, i32 }*
      %arraylen_ptr = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %90, i64 0, i32 1
      %arraylen = load i64, i64* %arraylen_ptr, align 8
      %inbounds = icmp ult i64 %89, %arraylen
      br i1 %inbounds, label %idxend, label %oob

L54:                                              ; preds = %L36
      %91 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 4
      %92 = bitcast {}** %91 to [3 x {}*]*
; └└└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:41 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:905 within `trunc`
    %ptls_field170177 = getelementptr inbounds {}**, {}*** %tls_pgcstack, i64 2
    %93 = bitcast {}*** %ptls_field170177 to i8**
    %ptls_load171178179 = load i8*, i8** %93, align 8
    %box = call noalias nonnull dereferenceable(16) {}* @ijl_gc_pool_alloc(i8* %ptls_load171178179, i32 752, i32 16) #14
    %94 = bitcast {}* %box to i64*
    %95 = getelementptr inbounds i64, i64* %94, i64 -1
    store atomic i64 139635484756704, i64* %95 unordered, align 8
    %96 = bitcast {}* %box to double*
    store double %65, double* %96, align 8
    %97 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %box, {}** %97, align 8
    call void @j_InexactError_3854([3 x {}*]* noalias nocapture noundef nonnull sret([3 x {}*]) %92, {}* inttoptr (i64 139635639651088 to {}*), {}* readonly inttoptr (i64 139635484757504 to {}*), {}* nonnull readonly %box)
    %ptls_load174180181 = load i8*, i8** %93, align 8
    %box114 = call noalias nonnull dereferenceable(32) {}* @ijl_gc_pool_alloc(i8* %ptls_load174180181, i32 800, i32 32) #14
    %98 = bitcast {}* %box114 to i64*
    %99 = getelementptr inbounds i64, i64* %98, i64 -1
    store atomic i64 139635415075680, i64* %99 unordered, align 8
    %100 = bitcast {}* %box114 to i8*
    %101 = bitcast {}** %91 to i8*
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %100, i8* noundef nonnull align 16 dereferenceable(24) %101, i64 24, i1 false)
    call void @ijl_throw({}* %box114)
    unreachable

L92:                                              ; preds = %pass
    %102 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %82, {}** %102, align 8
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
     %103 = call nonnull {}* @j_HIPStream_3851({}* inttoptr (i64 139635639629288 to {}*))
; │││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
      %getfield33 = load atomic {}*, {}** %87 unordered, align 8
; │││└
; │││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
      %104 = bitcast {}* %getfield33 to { i8*, i64, i16, i16, i32 }*
      %arraylen_ptr34 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %104, i64 0, i32 1
      %arraylen35 = load i64, i64* %arraylen_ptr34, align 8
      %inbounds36 = icmp ult i64 %89, %arraylen35
      br i1 %inbounds36, label %idxend39, label %oob37

L104:                                             ; preds = %pass107
; │││└
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
     store {}* inttoptr (i64 139631613718912 to {}*), {}** %.sub, align 8
     %105 = getelementptr inbounds [2 x {}*], [2 x {}*]* %12, i64 0, i64 1
     store {}* inttoptr (i64 139635639046152 to {}*), {}** %105, align 8
     %106 = call nonnull {}* @ijl_apply_generic({}* inttoptr (i64 139635413651600 to {}*), {}** nonnull %.sub, i32 2)
     call void @llvm.trap()
     unreachable

L109:                                             ; preds = %pass107, %142, %136, %merge_own
     %value_phi42 = phi {}* [ %arrayref, %pass107 ], [ %103, %142 ], [ %103, %136 ], [ %103, %merge_own ]
; ││└
; ││┌ @ iterators.jl:279 within `pairs`
; │││┌ @ essentials.jl:343 within `Pairs`
      %unbox45.unpack = load {}*, {}** inttoptr (i64 139631460108336 to {}**), align 16
      %unbox45.unpack145 = load {}*, {}** inttoptr (i64 139631460108344 to {}**), align 8
; ││└└
    %.fca.0.0.0.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 0, i64 0
    store i64 %34, i64* %.fca.0.0.0.gep, align 8
    %.fca.0.0.1.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 0, i64 1
    store i64 %35, i64* %.fca.0.0.1.gep, align 8
    %.fca.0.0.2.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 0, i64 2
    store i64 1, i64* %.fca.0.0.2.gep, align 8
    %.fca.0.1.0.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 1, i64 0
    store i64 %44, i64* %.fca.0.1.0.gep, align 8
    %.fca.0.1.1.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 1, i64 1
    store i64 %64, i64* %.fca.0.1.1.gep, align 8
    %.fca.0.1.2.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 1, i64 2
    store i64 %81, i64* %.fca.0.1.2.gep, align 8
    %.fca.1.0.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 1, i64 0
    store {}* %unbox45.unpack, {}** %14, align 16
    store {}* %unbox45.unpack, {}** %.fca.1.0.gep, align 8
    %.fca.1.1.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 1, i64 1
    store {}* %unbox45.unpack145, {}** %15, align 8
    store {}* %unbox45.unpack145, {}** %.fca.1.1.gep, align 8
    %107 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %value_phi42, {}** %107, align 8
    %108 = call nonnull {}* @"j_#_#15_3852"({}* nonnull %value_phi42, { [2 x [3 x i64]], [2 x {}*] }* nocapture readonly %24, { { i64, {}*, {}* } }* nocapture readonly %23, {}* readonly inttoptr (i64 139631800075624 to {}*), {}* nonnull %1, {}* nonnull %2, {}* nonnull %3, {}* nonnull %4, {}* nonnull %5, double %6, double %7, double %8, double %9, double %10, double %11)
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:45 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49 within `synchronize`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream`
    %109 = call nonnull {}* @"j_task_local_state!_3850"()
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:62
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
     %110 = bitcast {}* %109 to { i64, i32, {}*, i32 }*
     %111 = getelementptr inbounds { i64, i32, {}*, i32 }, { i64, i32, {}*, i32 }* %110, i64 0, i32 1
     %112 = load i32, i32* %111, align 4
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
     %113 = bitcast {}* %109 to i8*
     %getfield_addr52 = getelementptr inbounds i8, i8* %113, i64 40
     %114 = bitcast i8* %getfield_addr52 to {}**
     %getfield53 = load atomic {}*, {}** %114 unordered, align 8
; ││└
; ││┌ @ abstractarray.jl:1294 within `getindex`
; │││┌ @ indices.jl:350 within `to_indices` @ indices.jl:354
; ││││┌ @ indices.jl:359 within `_to_indices1`
; │││││┌ @ indices.jl:277 within `to_index` @ indices.jl:292
; ││││││┌ @ number.jl:7 within `convert`
; │││││││┌ @ boot.jl:784 within `Int64`
; ││││││││┌ @ boot.jl:703 within `toInt64`
           %115 = sext i32 %112 to i64
; │││└└└└└└
; │││ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
     %116 = add nsw i64 %115, -1
     %117 = bitcast {}* %getfield53 to { i8*, i64, i16, i16, i32 }*
     %arraylen_ptr54 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %117, i64 0, i32 1
     %arraylen55 = load i64, i64* %arraylen_ptr54, align 8
     %inbounds56 = icmp ult i64 %116, %arraylen55
     br i1 %inbounds56, label %idxend59, label %oob57

L123:                                             ; preds = %pass63
     store {}* %109, {}** %107, align 8
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
    %118 = call nonnull {}* @j_HIPStream_3851({}* inttoptr (i64 139635639629288 to {}*))
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
     %getfield66 = load atomic {}*, {}** %114 unordered, align 8
; ││└
; ││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
     %119 = bitcast {}* %getfield66 to { i8*, i64, i16, i16, i32 }*
     %arraylen_ptr67 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %119, i64 0, i32 1
     %arraylen68 = load i64, i64* %arraylen_ptr67, align 8
     %inbounds69 = icmp ult i64 %116, %arraylen68
     br i1 %inbounds69, label %idxend72, label %oob70

L135:                                             ; preds = %pass94
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
    store {}* inttoptr (i64 139631613718912 to {}*), {}** %.sub, align 8
    %120 = getelementptr inbounds [2 x {}*], [2 x {}*]* %12, i64 0, i64 1
    store {}* inttoptr (i64 139635639046152 to {}*), {}** %120, align 8
    %121 = call nonnull {}* @ijl_apply_generic({}* inttoptr (i64 139635413651600 to {}*), {}** nonnull %.sub, i32 2)
    call void @llvm.trap()
    unreachable

L140:                                             ; preds = %pass94, %161, %155, %merge_own77
    %value_phi82 = phi {}* [ %arrayref64, %pass94 ], [ %118, %161 ], [ %118, %155 ], [ %118, %merge_own77 ]
    store {}* %value_phi82, {}** %107, align 8
; │└
; │ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49 within `synchronize` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49
   call void @"j_#synchronize#31_3853"(i8 zeroext 0, i8 zeroext 0, {}* nonnull %value_phi82)
   %122 = load {}*, {}** %27, align 8
   %123 = bitcast {}*** %tls_pgcstack to {}**
   store {}* %122, {}** %123, align 8
   ret void

oob:                                              ; preds = %L52
; └
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; │││┌ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
      %errorbox = alloca i64, align 8
      store i64 %88, i64* %errorbox, align 8
      call void @ijl_bounds_error_ints({}* %getfield31, i64* nonnull %errorbox, i64 1)
      unreachable

idxend:                                           ; preds = %L52
      %124 = bitcast {}* %getfield31 to {}***
      %arrayptr143 = load {}**, {}*** %124, align 8
      %125 = getelementptr inbounds {}*, {}** %arrayptr143, i64 %89
      %arrayref = load {}*, {}** %125, align 8
      %.not = icmp eq {}* %arrayref, null
      br i1 %.not, label %fail, label %pass

fail:                                             ; preds = %idxend
      call void @ijl_throw({}* inttoptr (i64 139635483614496 to {}*))
      unreachable

pass:                                             ; preds = %idxend
; │││└
     %.not144 = icmp eq {}* %arrayref, inttoptr (i64 139635639046152 to {}*)
     br i1 %.not144, label %L92, label %pass107

oob37:                                            ; preds = %L92
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
; │││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
      %errorbox38 = alloca i64, align 8
      store i64 %88, i64* %errorbox38, align 8
      call void @ijl_bounds_error_ints({}* %getfield33, i64* nonnull %errorbox38, i64 1)
      unreachable

idxend39:                                         ; preds = %L92
      %arrayflags_ptr = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %104, i64 0, i32 2
      %arrayflags = load i16, i16* %arrayflags_ptr, align 2
      %126 = and i16 %arrayflags, 3
      %has_owner = icmp eq i16 %126, 3
      br i1 %has_owner, label %array_owned, label %merge_own

array_owned:                                      ; preds = %idxend39
      %127 = bitcast {}* %getfield33 to {}**
      %128 = getelementptr inbounds {}*, {}** %127, i64 5
      %external_owner = load {}*, {}** %128, align 8
      br label %merge_own

merge_own:                                        ; preds = %array_owned, %idxend39
      %data_owner = phi {}* [ %getfield33, %idxend39 ], [ %external_owner, %array_owned ]
      %129 = bitcast {}* %getfield33 to {}***
      %arrayptr41 = load {}**, {}*** %129, align 8
      %130 = getelementptr inbounds {}*, {}** %arrayptr41, i64 %89
      store atomic {}* %103, {}** %130 release, align 8
      %131 = bitcast {}* %data_owner to i64*
      %132 = getelementptr inbounds i64, i64* %131, i64 -1
      %133 = load atomic i64, i64* %132 unordered, align 8
      %134 = and i64 %133, 3
      %135 = icmp eq i64 %134, 3
      br i1 %135, label %136, label %L109

136:                                              ; preds = %merge_own
      %137 = bitcast {}* %103 to i64*
      %138 = getelementptr inbounds i64, i64* %137, i64 -1
      %139 = load atomic i64, i64* %138 unordered, align 8
      %140 = and i64 %139, 1
      %141 = icmp eq i64 %140, 0
      br i1 %141, label %142, label %L109

142:                                              ; preds = %136
      call void @ijl_gc_queue_root({}* nonnull %data_owner)
      br label %L109

oob57:                                            ; preds = %L109
; └└└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:45 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49 within `synchronize`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; ││┌ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
     %errorbox58 = alloca i64, align 8
     store i64 %115, i64* %errorbox58, align 8
     call void @ijl_bounds_error_ints({}* %getfield53, i64* nonnull %errorbox58, i64 1)
     unreachable

idxend59:                                         ; preds = %L109
     %143 = bitcast {}* %getfield53 to {}***
     %arrayptr61147 = load {}**, {}*** %143, align 8
     %144 = getelementptr inbounds {}*, {}** %arrayptr61147, i64 %116
     %arrayref64 = load {}*, {}** %144, align 8
     %.not148 = icmp eq {}* %arrayref64, null
     br i1 %.not148, label %fail62, label %pass63

fail62:                                           ; preds = %idxend59
     call void @ijl_throw({}* inttoptr (i64 139635483614496 to {}*))
     unreachable

pass63:                                           ; preds = %idxend59
; ││└
    %.not149 = icmp eq {}* %arrayref64, inttoptr (i64 139635639046152 to {}*)
    br i1 %.not149, label %L123, label %pass94

oob70:                                            ; preds = %L123
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
; ││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
     %errorbox71 = alloca i64, align 8
     store i64 %115, i64* %errorbox71, align 8
     call void @ijl_bounds_error_ints({}* %getfield66, i64* nonnull %errorbox71, i64 1)
     unreachable

idxend72:                                         ; preds = %L123
     %arrayflags_ptr73 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %119, i64 0, i32 2
     %arrayflags74 = load i16, i16* %arrayflags_ptr73, align 2
     %145 = and i16 %arrayflags74, 3
     %has_owner75 = icmp eq i16 %145, 3
     br i1 %has_owner75, label %array_owned76, label %merge_own77

array_owned76:                                    ; preds = %idxend72
     %146 = bitcast {}* %getfield66 to {}**
     %147 = getelementptr inbounds {}*, {}** %146, i64 5
     %external_owner78 = load {}*, {}** %147, align 8
     br label %merge_own77

merge_own77:                                      ; preds = %array_owned76, %idxend72
     %data_owner79 = phi {}* [ %getfield66, %idxend72 ], [ %external_owner78, %array_owned76 ]
     %148 = bitcast {}* %getfield66 to {}***
     %arrayptr81 = load {}**, {}*** %148, align 8
     %149 = getelementptr inbounds {}*, {}** %arrayptr81, i64 %116
     store atomic {}* %118, {}** %149 release, align 8
     %150 = bitcast {}* %data_owner79 to i64*
     %151 = getelementptr inbounds i64, i64* %150, i64 -1
     %152 = load atomic i64, i64* %151 unordered, align 8
     %153 = and i64 %152, 3
     %154 = icmp eq i64 %153, 3
     br i1 %154, label %155, label %L140

155:                                              ; preds = %merge_own77
     %156 = bitcast {}* %118 to i64*
     %157 = getelementptr inbounds i64, i64* %156, i64 -1
     %158 = load atomic i64, i64* %157 unordered, align 8
     %159 = and i64 %158, 1
     %160 = icmp eq i64 %159, 0
     br i1 %160, label %161, label %L140

161:                                              ; preds = %155
     call void @ijl_gc_queue_root({}* nonnull %data_owner79)
     br label %L140

pass94:                                           ; preds = %pass63
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
    %162 = bitcast {}* %arrayref64 to i64*
    %163 = getelementptr inbounds i64, i64* %162, i64 -1
    %164 = load atomic i64, i64* %163 unordered, align 8
    %165 = and i64 %164, -16
    %166 = inttoptr i64 %165 to {}*
    %exactly_isa.not = icmp eq {}* %166, inttoptr (i64 139631613718912 to {}*)
    br i1 %exactly_isa.not, label %L140, label %L135

pass107:                                          ; preds = %pass
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
     %167 = bitcast {}* %arrayref to i64*
     %168 = getelementptr inbounds i64, i64* %167, i64 -1
     %169 = load atomic i64, i64* %168 unordered, align 8
     %170 = and i64 %169, -16
     %171 = inttoptr i64 %170 to {}*
     %exactly_isa110.not = icmp eq {}* %171, inttoptr (i64 139631613718912 to {}*)
     br i1 %exactly_isa110.not, label %L109, label %L104
; └└└
}
Theoretical fetch size (GB): 2.1473856000000002
Theoretical write size (GB):2.1224160000000003
Laplacian kernel took: 233.921135 ms effective memory bandwidth: 18.253167248012883 GB/s
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:32 within `parallel_for`
define void @julia_parallel_for_3862([3 x i64]* nocapture noundef nonnull readonly align 8 dereferenceable(24) %0, {}* noundef nonnull align 8 dereferenceable(40) %1, {}* noundef nonnull align 8 dereferenceable(40) %2, {}* noundef nonnull align 8 dereferenceable(40) %3, {}* noundef nonnull align 8 dereferenceable(40) %4, {}* noundef nonnull align 8 dereferenceable(24) %5, double %6, double %7, double %8, double %9, double %10, double %11) #0 {
top:
  %12 = alloca [2 x {}*], align 8
  %gcframe176 = alloca [16 x {}*], align 16
  %gcframe176.sub = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 0
  %.sub = getelementptr inbounds [2 x {}*], [2 x {}*]* %12, i64 0, i64 0
  %13 = bitcast [16 x {}*]* %gcframe176 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %13, i8 0, i64 128, i1 true)
  %14 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 14
  %15 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 13
  %16 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 2
  %17 = bitcast {}** %16 to [2 x {}*]*
  %18 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %19 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %20 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %21 = alloca { [3 x i64], i8 addrspace(1)*, i64 }, align 8
  %22 = alloca { [1 x i64], i8 addrspace(1)*, i64 }, align 8
  %23 = alloca { { i64, {}*, {}* } }, align 8
  %24 = alloca { [2 x [3 x i64]], [2 x {}*] }, align 8
  %thread_ptr = call i8* asm "movq %fs:0, $0", "=r"() #16
  %tls_ppgcstack = getelementptr i8, i8* %thread_ptr, i64 -8
  %25 = bitcast i8* %tls_ppgcstack to {}****
  %tls_pgcstack = load {}***, {}**** %25, align 8
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:35 within `parallel_for`
; ┌ @ tuple.jl:92 within `indexed_iterate` @ tuple.jl:92
   %26 = bitcast [16 x {}*]* %gcframe176 to i64*
   store i64 56, i64* %26, align 16
   %27 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 1
   %28 = bitcast {}** %27 to {}***
   %29 = load {}**, {}*** %tls_pgcstack, align 8
   store {}** %29, {}*** %28, align 8
   %30 = bitcast {}*** %tls_pgcstack to {}***
   store {}** %gcframe176.sub, {}*** %30, align 8
   %31 = getelementptr inbounds [3 x i64], [3 x i64]* %0, i64 0, i64 0
; │ @ tuple.jl:92 within `indexed_iterate`
   %32 = getelementptr inbounds [3 x i64], [3 x i64]* %0, i64 0, i64 1
   %33 = getelementptr inbounds [3 x i64], [3 x i64]* %0, i64 0, i64 2
; └
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:36 within `parallel_for`
; ┌ @ promotion.jl:533 within `min`
; │┌ @ int.jl:83 within `<`
    %unbox = load i64, i64* %31, align 8
; │└
; │┌ @ essentials.jl:647 within `ifelse`
    %34 = call i64 @llvm.smin.i64(i64 %unbox, i64 32)
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:37 within `parallel_for`
; ┌ @ promotion.jl:533 within `min`
; │┌ @ int.jl:83 within `<`
    %unbox2 = load i64, i64* %32, align 8
; │└
; │┌ @ essentials.jl:647 within `ifelse`
    %35 = call i64 @llvm.smin.i64(i64 %unbox2, i64 32)
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:39 within `parallel_for`
; ┌ @ int.jl:97 within `/`
; │┌ @ float.jl:294 within `float`
; ││┌ @ float.jl:268 within `AbstractFloat`
; │││┌ @ float.jl:159 within `Float64`
      %36 = sitofp i64 %unbox to double
      %37 = sitofp i64 %34 to double
; │└└└
; │ @ int.jl:97 within `/` @ float.jl:412
   %38 = fdiv double %36, %37
; └
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:395 within `round`
    %39 = call double @llvm.ceil.f64(double %38)
; │└
; │┌ @ float.jl:902 within `trunc`
; ││┌ @ float.jl:537 within `<=`
     %40 = fcmp ult double %39, 0xC3E0000000000000
; ││└
    %41 = fcmp uge double %39, 0x43E0000000000000
    %42 = or i1 %40, %41
    br i1 %42, label %L21, label %L19

L19:                                              ; preds = %top
; ││ @ float.jl:903 within `trunc`
; ││┌ @ float.jl:336 within `unsafe_trunc`
     %43 = fptosi double %39 to i64
     %44 = freeze i64 %43
; └└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:40 within `parallel_for`
; ┌ @ int.jl:97 within `/`
; │┌ @ float.jl:294 within `float`
; ││┌ @ float.jl:268 within `AbstractFloat`
; │││┌ @ float.jl:159 within `Float64`
      %45 = sitofp i64 %unbox2 to double
      %46 = sitofp i64 %35 to double
; │└└└
; │ @ int.jl:97 within `/` @ float.jl:412
   %47 = fdiv double %45, %46
; └
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:395 within `round`
    %48 = call double @llvm.ceil.f64(double %47)
; │└
; │┌ @ float.jl:902 within `trunc`
; ││┌ @ float.jl:537 within `<=`
     %49 = fcmp ult double %48, 0xC3E0000000000000
; ││└
    %50 = fcmp uge double %48, 0x43E0000000000000
    %51 = or i1 %49, %50
    br i1 %51, label %L38, label %L36

L21:                                              ; preds = %top
    %52 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 7
    %53 = bitcast {}** %52 to [3 x {}*]*
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:39 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:905 within `trunc`
    %ptls_field187 = getelementptr inbounds {}**, {}*** %tls_pgcstack, i64 2
    %54 = bitcast {}*** %ptls_field187 to i8**
    %ptls_load188189 = load i8*, i8** %54, align 8
    %box123 = call noalias nonnull dereferenceable(16) {}* @ijl_gc_pool_alloc(i8* %ptls_load188189, i32 752, i32 16) #14
    %55 = bitcast {}* %box123 to i64*
    %56 = getelementptr inbounds i64, i64* %55, i64 -1
    store atomic i64 139635484756704, i64* %56 unordered, align 8
    %57 = bitcast {}* %box123 to double*
    store double %39, double* %57, align 8
    %58 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %box123, {}** %58, align 8
    call void @j_InexactError_3871([3 x {}*]* noalias nocapture noundef nonnull sret([3 x {}*]) %53, {}* inttoptr (i64 139635639651088 to {}*), {}* readonly inttoptr (i64 139635484757504 to {}*), {}* nonnull readonly %box123)
    %ptls_load162190191 = load i8*, i8** %54, align 8
    %box125 = call noalias nonnull dereferenceable(32) {}* @ijl_gc_pool_alloc(i8* %ptls_load162190191, i32 800, i32 32) #14
    %59 = bitcast {}* %box125 to i64*
    %60 = getelementptr inbounds i64, i64* %59, i64 -1
    store atomic i64 139635415075680, i64* %60 unordered, align 8
    %61 = bitcast {}* %box125 to i8*
    %62 = bitcast {}** %52 to i8*
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %61, i8* noundef nonnull align 8 dereferenceable(24) %62, i64 24, i1 false)
    call void @ijl_throw({}* %box125)
    unreachable

L36:                                              ; preds = %L19
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:40 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:903 within `trunc`
; ││┌ @ float.jl:336 within `unsafe_trunc`
     %63 = fptosi double %48 to i64
     %64 = freeze i64 %63
; └└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:41 within `parallel_for`
; ┌ @ int.jl:97 within `/`
; │┌ @ float.jl:294 within `float`
; ││┌ @ float.jl:268 within `AbstractFloat`
; │││┌ @ float.jl:159 within `Float64`
      %unbox8 = load i64, i64* %33, align 8
      %65 = sitofp i64 %unbox8 to double
; └└└└
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:902 within `trunc`
; ││┌ @ float.jl:537 within `<=`
     %66 = fcmp ult double %65, 0xC3E0000000000000
; ││└
    %67 = fcmp uge double %65, 0x43E0000000000000
    %68 = or i1 %66, %67
    br i1 %68, label %L54, label %L52

L38:                                              ; preds = %L19
    %69 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 10
    %70 = bitcast {}** %69 to [3 x {}*]*
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:40 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:905 within `trunc`
    %ptls_field164182 = getelementptr inbounds {}**, {}*** %tls_pgcstack, i64 2
    %71 = bitcast {}*** %ptls_field164182 to i8**
    %ptls_load165183184 = load i8*, i8** %71, align 8
    %box117 = call noalias nonnull dereferenceable(16) {}* @ijl_gc_pool_alloc(i8* %ptls_load165183184, i32 752, i32 16) #14
    %72 = bitcast {}* %box117 to i64*
    %73 = getelementptr inbounds i64, i64* %72, i64 -1
    store atomic i64 139635484756704, i64* %73 unordered, align 8
    %74 = bitcast {}* %box117 to double*
    store double %48, double* %74, align 8
    %75 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %box117, {}** %75, align 8
    call void @j_InexactError_3871([3 x {}*]* noalias nocapture noundef nonnull sret([3 x {}*]) %70, {}* inttoptr (i64 139635639651088 to {}*), {}* readonly inttoptr (i64 139635484757504 to {}*), {}* nonnull readonly %box117)
    %ptls_load168185186 = load i8*, i8** %71, align 8
    %box119 = call noalias nonnull dereferenceable(32) {}* @ijl_gc_pool_alloc(i8* %ptls_load168185186, i32 800, i32 32) #14
    %76 = bitcast {}* %box119 to i64*
    %77 = getelementptr inbounds i64, i64* %76, i64 -1
    store atomic i64 139635415075680, i64* %77 unordered, align 8
    %78 = bitcast {}* %box119 to i8*
    %79 = bitcast {}** %69 to i8*
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %78, i8* noundef nonnull align 16 dereferenceable(24) %79, i64 24, i1 false)
    call void @ijl_throw({}* %box119)
    unreachable

L52:                                              ; preds = %L36
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:41 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:903 within `trunc`
; ││┌ @ float.jl:336 within `unsafe_trunc`
     %80 = fptosi double %65 to i64
     %81 = freeze i64 %80
; └└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:170 within `macro expansion`
; │┌ @ tuple.jl:294 within `map` @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3864({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %18, {}* nonnull %1)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3864({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %19, {}* nonnull %2)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3864({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %20, {}* nonnull %3)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3864({ [3 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [3 x i64], i8 addrspace(1)*, i64 }) %21, {}* nonnull %4)
; ││└└└└
; ││ @ tuple.jl:294 within `map` @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294 @ tuple.jl:294
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:110 within `rocconvert`
; │││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:40 within `adapt`
; ││││┌ @ /home/y1e/.julia/packages/Adapt/7T9au/src/Adapt.jl:57 within `adapt_structure`
; │││││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/gpuarrays.jl:67 within `adapt_storage`
        call void @j_convert_3865({ [1 x i64], i8 addrspace(1)*, i64 }* noalias nocapture noundef nonnull sret({ [1 x i64], i8 addrspace(1)*, i64 }) %22, {}* nonnull %5)
; │└└└└└
; │ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:172 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/compiler/codegen.jl:132 within `hipfunction`
    call void @"j_#hipfunction#37_3866"({ { i64, {}*, {}* } }* noalias nocapture noundef nonnull sret({ { i64, {}*, {}* } }) %23, [2 x {}*]* noalias nocapture noundef nonnull %17)
; │└
; │ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream`
     %82 = call nonnull {}* @"j_task_local_state!_3867"()
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:62
; │││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
      %83 = bitcast {}* %82 to { i64, i32, {}*, i32 }*
      %84 = getelementptr inbounds { i64, i32, {}*, i32 }, { i64, i32, {}*, i32 }* %83, i64 0, i32 1
      %85 = load i32, i32* %84, align 4
; │││└
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; │││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
      %86 = bitcast {}* %82 to i8*
      %getfield_addr30 = getelementptr inbounds i8, i8* %86, i64 40
      %87 = bitcast i8* %getfield_addr30 to {}**
      %getfield31 = load atomic {}*, {}** %87 unordered, align 8
; │││└
; │││┌ @ abstractarray.jl:1294 within `getindex`
; ││││┌ @ indices.jl:350 within `to_indices` @ indices.jl:354
; │││││┌ @ indices.jl:359 within `_to_indices1`
; ││││││┌ @ indices.jl:277 within `to_index` @ indices.jl:292
; │││││││┌ @ number.jl:7 within `convert`
; ││││││││┌ @ boot.jl:784 within `Int64`
; │││││││││┌ @ boot.jl:703 within `toInt64`
            %88 = sext i32 %85 to i64
; ││││└└└└└└
; ││││ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
      %89 = add nsw i64 %88, -1
      %90 = bitcast {}* %getfield31 to { i8*, i64, i16, i16, i32 }*
      %arraylen_ptr = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %90, i64 0, i32 1
      %arraylen = load i64, i64* %arraylen_ptr, align 8
      %inbounds = icmp ult i64 %89, %arraylen
      br i1 %inbounds, label %idxend, label %oob

L54:                                              ; preds = %L36
      %91 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 4
      %92 = bitcast {}** %91 to [3 x {}*]*
; └└└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:41 within `parallel_for`
; ┌ @ float.jl:384 within `ceil`
; │┌ @ float.jl:905 within `trunc`
    %ptls_field170177 = getelementptr inbounds {}**, {}*** %tls_pgcstack, i64 2
    %93 = bitcast {}*** %ptls_field170177 to i8**
    %ptls_load171178179 = load i8*, i8** %93, align 8
    %box = call noalias nonnull dereferenceable(16) {}* @ijl_gc_pool_alloc(i8* %ptls_load171178179, i32 752, i32 16) #14
    %94 = bitcast {}* %box to i64*
    %95 = getelementptr inbounds i64, i64* %94, i64 -1
    store atomic i64 139635484756704, i64* %95 unordered, align 8
    %96 = bitcast {}* %box to double*
    store double %65, double* %96, align 8
    %97 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %box, {}** %97, align 8
    call void @j_InexactError_3871([3 x {}*]* noalias nocapture noundef nonnull sret([3 x {}*]) %92, {}* inttoptr (i64 139635639651088 to {}*), {}* readonly inttoptr (i64 139635484757504 to {}*), {}* nonnull readonly %box)
    %ptls_load174180181 = load i8*, i8** %93, align 8
    %box114 = call noalias nonnull dereferenceable(32) {}* @ijl_gc_pool_alloc(i8* %ptls_load174180181, i32 800, i32 32) #14
    %98 = bitcast {}* %box114 to i64*
    %99 = getelementptr inbounds i64, i64* %98, i64 -1
    store atomic i64 139635415075680, i64* %99 unordered, align 8
    %100 = bitcast {}* %box114 to i8*
    %101 = bitcast {}** %91 to i8*
    call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %100, i8* noundef nonnull align 16 dereferenceable(24) %101, i64 24, i1 false)
    call void @ijl_throw({}* %box114)
    unreachable

L92:                                              ; preds = %pass
    %102 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %82, {}** %102, align 8
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
     %103 = call nonnull {}* @j_HIPStream_3868({}* inttoptr (i64 139635639629288 to {}*))
; │││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
      %getfield33 = load atomic {}*, {}** %87 unordered, align 8
; │││└
; │││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
      %104 = bitcast {}* %getfield33 to { i8*, i64, i16, i16, i32 }*
      %arraylen_ptr34 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %104, i64 0, i32 1
      %arraylen35 = load i64, i64* %arraylen_ptr34, align 8
      %inbounds36 = icmp ult i64 %89, %arraylen35
      br i1 %inbounds36, label %idxend39, label %oob37

L104:                                             ; preds = %pass107
; │││└
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
     store {}* inttoptr (i64 139631613718912 to {}*), {}** %.sub, align 8
     %105 = getelementptr inbounds [2 x {}*], [2 x {}*]* %12, i64 0, i64 1
     store {}* inttoptr (i64 139635639046152 to {}*), {}** %105, align 8
     %106 = call nonnull {}* @ijl_apply_generic({}* inttoptr (i64 139635413651600 to {}*), {}** nonnull %.sub, i32 2)
     call void @llvm.trap()
     unreachable

L109:                                             ; preds = %pass107, %142, %136, %merge_own
     %value_phi42 = phi {}* [ %arrayref, %pass107 ], [ %103, %142 ], [ %103, %136 ], [ %103, %merge_own ]
; ││└
; ││┌ @ iterators.jl:279 within `pairs`
; │││┌ @ essentials.jl:343 within `Pairs`
      %unbox45.unpack = load {}*, {}** inttoptr (i64 139631460108336 to {}**), align 16
      %unbox45.unpack145 = load {}*, {}** inttoptr (i64 139631460108344 to {}**), align 8
; ││└└
    %.fca.0.0.0.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 0, i64 0
    store i64 %34, i64* %.fca.0.0.0.gep, align 8
    %.fca.0.0.1.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 0, i64 1
    store i64 %35, i64* %.fca.0.0.1.gep, align 8
    %.fca.0.0.2.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 0, i64 2
    store i64 1, i64* %.fca.0.0.2.gep, align 8
    %.fca.0.1.0.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 1, i64 0
    store i64 %44, i64* %.fca.0.1.0.gep, align 8
    %.fca.0.1.1.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 1, i64 1
    store i64 %64, i64* %.fca.0.1.1.gep, align 8
    %.fca.0.1.2.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 0, i64 1, i64 2
    store i64 %81, i64* %.fca.0.1.2.gep, align 8
    %.fca.1.0.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 1, i64 0
    store {}* %unbox45.unpack, {}** %14, align 16
    store {}* %unbox45.unpack, {}** %.fca.1.0.gep, align 8
    %.fca.1.1.gep = getelementptr inbounds { [2 x [3 x i64]], [2 x {}*] }, { [2 x [3 x i64]], [2 x {}*] }* %24, i64 0, i32 1, i64 1
    store {}* %unbox45.unpack145, {}** %15, align 8
    store {}* %unbox45.unpack145, {}** %.fca.1.1.gep, align 8
    %107 = getelementptr inbounds [16 x {}*], [16 x {}*]* %gcframe176, i64 0, i64 15
    store {}* %value_phi42, {}** %107, align 8
    %108 = call nonnull {}* @"j_#_#15_3869"({}* nonnull %value_phi42, { [2 x [3 x i64]], [2 x {}*] }* nocapture readonly %24, { { i64, {}*, {}* } }* nocapture readonly %23, {}* readonly inttoptr (i64 139631800075624 to {}*), {}* nonnull %1, {}* nonnull %2, {}* nonnull %3, {}* nonnull %4, {}* nonnull %5, double %6, double %7, double %8, double %9, double %10, double %11)
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:45 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49 within `synchronize`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream`
    %109 = call nonnull {}* @"j_task_local_state!_3867"()
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:62
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
     %110 = bitcast {}* %109 to { i64, i32, {}*, i32 }*
     %111 = getelementptr inbounds { i64, i32, {}*, i32 }, { i64, i32, {}*, i32 }* %110, i64 0, i32 1
     %112 = load i32, i32* %111, align 4
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
     %113 = bitcast {}* %109 to i8*
     %getfield_addr52 = getelementptr inbounds i8, i8* %113, i64 40
     %114 = bitcast i8* %getfield_addr52 to {}**
     %getfield53 = load atomic {}*, {}** %114 unordered, align 8
; ││└
; ││┌ @ abstractarray.jl:1294 within `getindex`
; │││┌ @ indices.jl:350 within `to_indices` @ indices.jl:354
; ││││┌ @ indices.jl:359 within `_to_indices1`
; │││││┌ @ indices.jl:277 within `to_index` @ indices.jl:292
; ││││││┌ @ number.jl:7 within `convert`
; │││││││┌ @ boot.jl:784 within `Int64`
; ││││││││┌ @ boot.jl:703 within `toInt64`
           %115 = sext i32 %112 to i64
; │││└└└└└└
; │││ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
     %116 = add nsw i64 %115, -1
     %117 = bitcast {}* %getfield53 to { i8*, i64, i16, i16, i32 }*
     %arraylen_ptr54 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %117, i64 0, i32 1
     %arraylen55 = load i64, i64* %arraylen_ptr54, align 8
     %inbounds56 = icmp ult i64 %116, %arraylen55
     br i1 %inbounds56, label %idxend59, label %oob57

L123:                                             ; preds = %pass63
     store {}* %109, {}** %107, align 8
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
    %118 = call nonnull {}* @j_HIPStream_3868({}* inttoptr (i64 139635639629288 to {}*))
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:19 within `getproperty`
     %getfield66 = load atomic {}*, {}** %114 unordered, align 8
; ││└
; ││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
     %119 = bitcast {}* %getfield66 to { i8*, i64, i16, i16, i32 }*
     %arraylen_ptr67 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %119, i64 0, i32 1
     %arraylen68 = load i64, i64* %arraylen_ptr67, align 8
     %inbounds69 = icmp ult i64 %116, %arraylen68
     br i1 %inbounds69, label %idxend72, label %oob70

L135:                                             ; preds = %pass94
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
    store {}* inttoptr (i64 139631613718912 to {}*), {}** %.sub, align 8
    %120 = getelementptr inbounds [2 x {}*], [2 x {}*]* %12, i64 0, i64 1
    store {}* inttoptr (i64 139635639046152 to {}*), {}** %120, align 8
    %121 = call nonnull {}* @ijl_apply_generic({}* inttoptr (i64 139635413651600 to {}*), {}** nonnull %.sub, i32 2)
    call void @llvm.trap()
    unreachable

L140:                                             ; preds = %pass94, %161, %155, %merge_own77
    %value_phi82 = phi {}* [ %arrayref64, %pass94 ], [ %118, %161 ], [ %118, %155 ], [ %118, %merge_own77 ]
    store {}* %value_phi82, {}** %107, align 8
; │└
; │ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49 within `synchronize` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49
   call void @"j_#synchronize#31_3870"(i8 zeroext 0, i8 zeroext 0, {}* nonnull %value_phi82)
   %122 = load {}*, {}** %27, align 8
   %123 = bitcast {}*** %tls_pgcstack to {}**
   store {}* %122, {}** %123, align 8
   ret void

oob:                                              ; preds = %L52
; └
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; │││┌ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
      %errorbox = alloca i64, align 8
      store i64 %88, i64* %errorbox, align 8
      call void @ijl_bounds_error_ints({}* %getfield31, i64* nonnull %errorbox, i64 1)
      unreachable

idxend:                                           ; preds = %L52
      %124 = bitcast {}* %getfield31 to {}***
      %arrayptr143 = load {}**, {}*** %124, align 8
      %125 = getelementptr inbounds {}*, {}** %arrayptr143, i64 %89
      %arrayref = load {}*, {}** %125, align 8
      %.not = icmp eq {}* %arrayref, null
      br i1 %.not, label %fail, label %pass

fail:                                             ; preds = %idxend
      call void @ijl_throw({}* inttoptr (i64 139635483614496 to {}*))
      unreachable

pass:                                             ; preds = %idxend
; │││└
     %.not144 = icmp eq {}* %arrayref, inttoptr (i64 139635639046152 to {}*)
     br i1 %.not144, label %L92, label %pass107

oob37:                                            ; preds = %L92
; │││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
; │││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
      %errorbox38 = alloca i64, align 8
      store i64 %88, i64* %errorbox38, align 8
      call void @ijl_bounds_error_ints({}* %getfield33, i64* nonnull %errorbox38, i64 1)
      unreachable

idxend39:                                         ; preds = %L92
      %arrayflags_ptr = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %104, i64 0, i32 2
      %arrayflags = load i16, i16* %arrayflags_ptr, align 2
      %126 = and i16 %arrayflags, 3
      %has_owner = icmp eq i16 %126, 3
      br i1 %has_owner, label %array_owned, label %merge_own

array_owned:                                      ; preds = %idxend39
      %127 = bitcast {}* %getfield33 to {}**
      %128 = getelementptr inbounds {}*, {}** %127, i64 5
      %external_owner = load {}*, {}** %128, align 8
      br label %merge_own

merge_own:                                        ; preds = %array_owned, %idxend39
      %data_owner = phi {}* [ %getfield33, %idxend39 ], [ %external_owner, %array_owned ]
      %129 = bitcast {}* %getfield33 to {}***
      %arrayptr41 = load {}**, {}*** %129, align 8
      %130 = getelementptr inbounds {}*, {}** %arrayptr41, i64 %89
      store atomic {}* %103, {}** %130 release, align 8
      %131 = bitcast {}* %data_owner to i64*
      %132 = getelementptr inbounds i64, i64* %131, i64 -1
      %133 = load atomic i64, i64* %132 unordered, align 8
      %134 = and i64 %133, 3
      %135 = icmp eq i64 %134, 3
      br i1 %135, label %136, label %L109

136:                                              ; preds = %merge_own
      %137 = bitcast {}* %103 to i64*
      %138 = getelementptr inbounds i64, i64* %137, i64 -1
      %139 = load atomic i64, i64* %138 unordered, align 8
      %140 = and i64 %139, 1
      %141 = icmp eq i64 %140, 0
      br i1 %141, label %142, label %L109

142:                                              ; preds = %136
      call void @ijl_gc_queue_root({}* nonnull %data_owner)
      br label %L109

oob57:                                            ; preds = %L109
; └└└└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:45 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:49 within `synchronize`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:63
; ││┌ @ abstractarray.jl:1294 within `getindex` @ essentials.jl:13
     %errorbox58 = alloca i64, align 8
     store i64 %115, i64* %errorbox58, align 8
     call void @ijl_bounds_error_ints({}* %getfield53, i64* nonnull %errorbox58, i64 1)
     unreachable

idxend59:                                         ; preds = %L109
     %143 = bitcast {}* %getfield53 to {}***
     %arrayptr61147 = load {}**, {}*** %143, align 8
     %144 = getelementptr inbounds {}*, {}** %arrayptr61147, i64 %116
     %arrayref64 = load {}*, {}** %144, align 8
     %.not148 = icmp eq {}* %arrayref64, null
     br i1 %.not148, label %fail62, label %pass63

fail62:                                           ; preds = %idxend59
     call void @ijl_throw({}* inttoptr (i64 139635483614496 to {}*))
     unreachable

pass63:                                           ; preds = %idxend59
; ││└
    %.not149 = icmp eq {}* %arrayref64, inttoptr (i64 139635639046152 to {}*)
    br i1 %.not149, label %L123, label %pass94

oob70:                                            ; preds = %L123
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:64
; ││┌ @ multidimensional.jl:698 within `setindex!` @ array.jl:1021
     %errorbox71 = alloca i64, align 8
     store i64 %115, i64* %errorbox71, align 8
     call void @ijl_bounds_error_ints({}* %getfield66, i64* nonnull %errorbox71, i64 1)
     unreachable

idxend72:                                         ; preds = %L123
     %arrayflags_ptr73 = getelementptr inbounds { i8*, i64, i16, i16, i32 }, { i8*, i64, i16, i16, i32 }* %119, i64 0, i32 2
     %arrayflags74 = load i16, i16* %arrayflags_ptr73, align 2
     %145 = and i16 %arrayflags74, 3
     %has_owner75 = icmp eq i16 %145, 3
     br i1 %has_owner75, label %array_owned76, label %merge_own77

array_owned76:                                    ; preds = %idxend72
     %146 = bitcast {}* %getfield66 to {}**
     %147 = getelementptr inbounds {}*, {}** %146, i64 5
     %external_owner78 = load {}*, {}** %147, align 8
     br label %merge_own77

merge_own77:                                      ; preds = %array_owned76, %idxend72
     %data_owner79 = phi {}* [ %getfield66, %idxend72 ], [ %external_owner78, %array_owned76 ]
     %148 = bitcast {}* %getfield66 to {}***
     %arrayptr81 = load {}**, {}*** %148, align 8
     %149 = getelementptr inbounds {}*, {}** %arrayptr81, i64 %116
     store atomic {}* %118, {}** %149 release, align 8
     %150 = bitcast {}* %data_owner79 to i64*
     %151 = getelementptr inbounds i64, i64* %150, i64 -1
     %152 = load atomic i64, i64* %151 unordered, align 8
     %153 = and i64 %152, 3
     %154 = icmp eq i64 %153, 3
     br i1 %154, label %155, label %L140

155:                                              ; preds = %merge_own77
     %156 = bitcast {}* %118 to i64*
     %157 = getelementptr inbounds i64, i64* %156, i64 -1
     %158 = load atomic i64, i64* %157 unordered, align 8
     %159 = and i64 %158, 1
     %160 = icmp eq i64 %159, 0
     br i1 %160, label %161, label %L140

161:                                              ; preds = %155
     call void @ijl_gc_queue_root({}* nonnull %data_owner79)
     br label %L140

pass94:                                           ; preds = %pass63
; ││└
; ││ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
    %162 = bitcast {}* %arrayref64 to i64*
    %163 = getelementptr inbounds i64, i64* %162, i64 -1
    %164 = load atomic i64, i64* %163 unordered, align 8
    %165 = and i64 %164, -16
    %166 = inttoptr i64 %165 to {}*
    %exactly_isa.not = icmp eq {}* %166, inttoptr (i64 139631613718912 to {}*)
    br i1 %exactly_isa.not, label %L140, label %L135

pass107:                                          ; preds = %pass
; └└
;  @ /home/y1e/.julia/packages/JACC/CPpH7/ext/JACCAMDGPU/JACCAMDGPU.jl:42 within `parallel_for`
; ┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/highlevel.jl:175 within `macro expansion`
; │┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/runtime/hip-execution.jl:54 within `HIPKernel`
; ││┌ @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:76 within `stream` @ /home/y1e/.julia/packages/AMDGPU/gtxsf/src/tls.jl:66
     %167 = bitcast {}* %arrayref to i64*
     %168 = getelementptr inbounds i64, i64* %167, i64 -1
     %169 = load atomic i64, i64* %168 unordered, align 8
     %170 = and i64 %169, -16
     %171 = inttoptr i64 %170 to {}*
     %exactly_isa110.not = icmp eq {}* %171, inttoptr (i64 139631613718912 to {}*)
     br i1 %exactly_isa110.not, label %L109, label %L104
; └└└
}
Theoretical fetch size (GB): 2.1473856000000002
Theoretical write size (GB):2.1224160000000003
Laplacian kernel took: 13.314805000000002 ms effective memory bandwidth: 320.68074598163474 GB/s
 23.059182 seconds (9.86 M allocations: 4.686 GiB, 6.58% gc time, 76.91% compilation time: <1% of which was recompilation)
