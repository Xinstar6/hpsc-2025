	.file	"11_nbody.cpp"
	.text
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"%d %f %f\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB5691:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-64, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x70
	xorl	%ebx, %ebx
	subq	$544, %rsp
	.p2align 4
	.p2align 3
.L2:
	call	drand48
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, -368(%rbp,%rbx)
	call	drand48
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, -304(%rbp,%rbx)
	call	drand48
	movl	$0x00000000, -112(%rbp,%rbx)
	movl	$0x00000000, -176(%rbp,%rbx)
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, -240(%rbp,%rbx)
	addq	$4, %rbx
	cmpq	$64, %rbx
	jne	.L2
	vmovaps	-368(%rbp), %zmm7
	xorl	%ebx, %ebx
	vmovaps	%zmm7, -432(%rbp)
	vmovaps	-304(%rbp), %zmm7
	vmovaps	%zmm7, -496(%rbp)
	vmovaps	-240(%rbp), %zmm7
	vmovaps	%zmm7, -560(%rbp)
	.p2align 4
	.p2align 3
.L3:
	vbroadcastss	-368(%rbp,%rbx,4), %zmm5
	movl	%ebx, %esi
	movl	$.LC2, %edi
	movl	$2, %eax
	vsubps	-432(%rbp), %zmm5, %zmm0
	vbroadcastss	-304(%rbp,%rbx,4), %zmm7
	vsubps	-496(%rbp), %zmm7, %zmm1
	vmulps	%zmm0, %zmm0, %zmm2
	vfmadd231ps	%zmm1, %zmm1, %zmm2
	vrsqrt14ps	%zmm2, %zmm4
	vcmpps	$14, .LC1(%rip), %zmm2, %k1
	vmulps	%zmm4, %zmm4, %zmm2
	vmulps	%zmm4, %zmm2, %zmm3{%k1}{z}
	vmulps	-560(%rbp), %zmm3, %zmm3
	vmulps	%zmm3, %zmm0, %zmm0
	vmulps	%zmm3, %zmm1, %zmm1
	vextractf64x4	$0x1, %zmm0, %ymm2
	vaddps	%ymm0, %ymm2, %ymm0
	vextractf128	$0x1, %ymm0, %xmm2
	vaddps	%xmm0, %xmm2, %xmm2
	vpermilps	$78, %xmm2, %xmm0
	vaddps	%xmm2, %xmm0, %xmm0
	vmovaps	%xmm0, %xmm2
	vshufps	$85, %xmm0, %xmm0, %xmm0
	vaddss	%xmm0, %xmm2, %xmm2
	vmovss	-176(%rbp,%rbx,4), %xmm0
	vsubss	%xmm2, %xmm0, %xmm0
	vextractf64x4	$0x1, %zmm1, %ymm2
	vaddps	%ymm1, %ymm2, %ymm1
	vmovss	%xmm0, -176(%rbp,%rbx,4)
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vextractf128	$0x1, %ymm1, %xmm2
	vaddps	%xmm1, %xmm2, %xmm1
	vpermilps	$78, %xmm1, %xmm2
	vaddps	%xmm1, %xmm2, %xmm2
	vmovaps	%xmm2, %xmm1
	vshufps	$85, %xmm2, %xmm2, %xmm2
	vaddss	%xmm2, %xmm1, %xmm2
	vmovss	-112(%rbp,%rbx,4), %xmm1
	vsubss	%xmm2, %xmm1, %xmm1
	vmovss	%xmm1, -112(%rbp,%rbx,4)
	vcvtss2sd	%xmm1, %xmm1, %xmm1
	vzeroupper
	incq	%rbx
	call	printf
	cmpq	$16, %rbx
	jne	.L3
	addq	$544, %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r10
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5691:
	.size	main, .-main
	.section	.rodata
	.align 64
.LC1:
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.long	841731191
	.ident	"GCC: (GNU) 11.4.1 20231218 (Red Hat 11.4.1-3)"
	.section	.note.GNU-stack,"",@progbits
