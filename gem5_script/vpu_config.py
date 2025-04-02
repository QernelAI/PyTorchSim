import m5
from m5.objects import *

class SystolicArray(MinorFU):
    unitType = "SystolicArray"
    opClasses = minorMakeOpClassSet(["CustomMatMul", "CustomMatMuliVpush", "CustomMatMulwVpush", "CustomMatMulvpop"])
    opLat = 1
    systolicArrayWidth = 128
    systolicArrayHeight = 128

class SparseAccelerator(MinorFU):
    unitType = "SparseAccelerator"
    opClasses = minorMakeOpClassSet(["CustomMatMul", "CustomMatMuliVpush", "CustomMatMulwVpush", "CustomMatMulvpop"])
    opLat = 1

class SpecialFunctionUnit(MinorFU):
    opClasses = minorMakeOpClassSet([
        "CustomMatMulvexp",
        "CustomMatMulverf",
        "CustomMatMulvtanh",
        ])
    opLat = 10

class MinorFPUnit(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "FloatAdd",
            "FloatCmp",
            "FloatCvt",
            "FloatMult",
            "FloatMultAcc",
            "FloatDiv",
            "FloatMisc",
            "FloatSqrt"
        ]
    )

class MinorVecAdder(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdAdd",
            "SimdFloatAdd",
            "SimdFloatAlu",
            "SimdFloatCmp",
            "SimdShift",
            "SimdShiftAcc",
            "SimdAddAcc",
            "SimdAlu",
            "SimdCmp",
        ]
    )
    opLat = 1

class MinorVecMultiplier(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdMult",
            "SimdFloatMult",
            "SimdMultAcc",
            "SimdMatMultAcc",
            "SimdSqrt",
            "SimdFloatMultAcc",
            "SimdFloatMatMultAcc",
            "SimdFloatSqrt",
        ]
    )
    opLat = 1

class MinorVecDivider(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdDiv",
            "SimdFloatDiv",
        ]
    )
    opLat = 1

class MinorVecReduce(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdReduceAdd",
            "SimdReduceAlu",
            "SimdReduceCmp",
            "SimdFloatReduceAdd",
            "SimdFloatReduceCmp",
        ]
    )
    opLat = 1

class MinorVecLdStore(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdUnitStrideLoad",
            "SimdUnitStrideStore",
            "SimdUnitStrideMaskLoad",
            "SimdUnitStrideMaskStore",
            "SimdStridedLoad",
            "SimdStridedStore",
            "SimdIndexedLoad",
            "SimdIndexedStore",
            "SimdUnitStrideFaultOnlyFirstLoad",
            "SimdWholeRegisterLoad",
            "SimdWholeRegisterStore",
            "SimdUnitStrideSegmentedLoad",
            "SimdUnitStrideSegmentedStore",
        ]
    )
    opLat = 1

class MinorVecMisc(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdCvt",
            "SimdFloatCvt",
            "SimdFloatMisc",
            "SimdPredAlu",
            "SimdMisc",
            "SimdExt",
            "SimdFloatExt",
            "CustomVlaneIdx",
        ]
    )
    opLat = 1

class MinorVecConfig(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdConfig",
        ]
    )
    opLat = 1

class MinorCustomIntFU(MinorDefaultIntFU):
    opLat = 1

class MinorCustomIntDivFU(MinorDefaultIntDivFU):
    opLat = 1

class MinorCustomIntMulFU(MinorDefaultIntMulFU):
    opLat = 1

class MinorCustomPredFU(MinorDefaultPredFU):
    opLat = 1

class MinorCustomMemFU(MinorDefaultMemFU):
    opLat = 1

class MinorCustomMiscFU(MinorDefaultMiscFU):
    opLat = 1

class MinorCustomFUPool(MinorFUPool):
    funcUnits = [
        # Scalar unit
        MinorFPUnit(),
        MinorCustomIntFU(),
        MinorCustomIntFU(),
        MinorCustomIntMulFU(),
        MinorCustomIntDivFU(),
        MinorCustomPredFU(),
        MinorCustomMemFU(),
        MinorCustomMiscFU(),

        # Matmul unit
        SystolicArray(), # 0
 
        # Vector
        MinorVecConfig(), # 1 for vector config
        MinorVecConfig(),
        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecLdStore(),
        MinorVecLdStore(),

        # Vector ALU0
        MinorVecAdder(), # 6
        MinorVecMultiplier(), # 7
        MinorVecDivider(), # 8
        MinorVecReduce(),

        # Vector ALU1
        MinorVecAdder(), # 18 ~ 29
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecReduce(),

        # SFU
        SpecialFunctionUnit(),
    ]

class RiscvVPU(RiscvMinorCPU):
    fetch2InputBufferSize = 2
    decodeInputBufferSize = 1
    decodeInputWidth = 1
    executeInputWidth = 8
    executeIssueLimit = 8
    executeCommitLimit = 8
    # Memory
    executeMemoryIssueLimit = 2
    executeMemoryCommitLimit = 2
    executeMaxAccessesInMemory = 2
    executeLSQMaxStoreBufferStoresPerCycle = 2
    executeLSQTransfersQueueSize = 8
    executeLSQStoreBufferSize = 8

    executeFuncUnits = MinorCustomFUPool()
