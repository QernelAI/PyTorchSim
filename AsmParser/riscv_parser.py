from collections import OrderedDict
from itertools import chain
import onnx
from onnx_utility import loop_index_node, load_node, store_node, compute_node, connect_nodes, dump_onnx_graph

# Operand Attributes
MEM =       0x400
OFFSET =    0x200
DEST =      0x100

REGISTER =  0x000
IMMEDIATE = 0x001
SPECIAL =   0x002
LABEL =     0x004
TYPE_MASK = (IMMEDIATE|SPECIAL|LABEL)

R_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER]
I_TEMPLATE = [REGISTER|DEST, REGISTER, IMMEDIATE]
U_TEMPLATE = [REGISTER|DEST, IMMEDIATE]
B_TEMPLATE = [REGISTER, REGISTER, LABEL]

LOAD_TEMPLATE = [REGISTER|DEST, REGISTER|OFFSET]
STORE_TEMPLATE = [REGISTER, REGISTER|OFFSET]

NOP_TEMPLATE = []
RI_TEMPLATE = [REGISTER|DEST, IMMEDIATE]
PSEUDO_B_TEMPLATE = [REGISTER, LABEL]
PSEUDO_J_TEMPLATE = [LABEL]

CSR_REG_TEMPLATE = [REGISTER|DEST, SPECIAL, REGISTER]
CSR_IMM_TEMPLATE = [REGISTER|DEST, SPECIAL, IMMEDIATE]

CSR_ALIAS_WRITE_TEMPLATE = [SPECIAL, REGISTER]
CSR_ALIAS_READ_TEMPLATE = [REGISTER|DEST, SPECIAL]
CSR_ALIAS_IMM_TEPLATE = [SPECIAL, IMMEDIATE]

ATOMIC_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER|MEM]
ATOMIC_LOAD_TEMPLATE = [REGISTER|DEST, REGISTER|MEM]

R4_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER, REGISTER]
R2_TEMPLATE = [REGISTER|DEST, REGISTER]

VECTOR_LOAD_TEMPLATE = [REGISTER|DEST, REGISTER|MEM]
VECTOR_STORE_TEMPLATE = [REGISTER, REGISTER|MEM]

VECTOR_STRIDE_LOAD_TEMPLATE = [REGISTER|DEST, REGISTER|MEM, REGISTER]
VECTOR_STRIDE_STORE_TEMPLATE = [REGISTER, REGISTER|MEM, REGISTER]

VECTOR_VV_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER]
VECTOR_VX_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER]
VECTOR_VF_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER]
VECTOR_VI_TEMPLATE = [REGISTER|DEST, REGISTER, IMMEDIATE]
VECTOR_WV_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER]
VECTOR_WX_TEMPLATE = [REGISTER|DEST, REGISTER, REGISTER]

R32_INSTUCTION_TEMPLATE = {
    # RV32
    "addi": I_TEMPLATE,
    "slti": I_TEMPLATE,
    "sltiu": I_TEMPLATE,
    "xori": I_TEMPLATE,
    "ori": I_TEMPLATE,
    "andi": I_TEMPLATE,
    "lui": I_TEMPLATE,
    "slli": I_TEMPLATE,
    "srli": I_TEMPLATE,
    "srai": I_TEMPLATE,

    "add": R_TEMPLATE,
    "sub": R_TEMPLATE,
    "slt": R_TEMPLATE,
    "sltu": R_TEMPLATE,
    "xor": R_TEMPLATE,
    "or": R_TEMPLATE,
    "and": R_TEMPLATE,
    "sll": R_TEMPLATE,
    "srl": R_TEMPLATE,
    "sra": R_TEMPLATE,

    "beq": B_TEMPLATE,
    "bne": B_TEMPLATE,
    "blt": B_TEMPLATE,
    "bge": B_TEMPLATE,
    "bltu": B_TEMPLATE,
    "bgeu": B_TEMPLATE,

    "lb": LOAD_TEMPLATE,
    "lh": LOAD_TEMPLATE,
    "lw": LOAD_TEMPLATE,
    "lbu": LOAD_TEMPLATE,
    "lhu": LOAD_TEMPLATE,
    "sb": STORE_TEMPLATE,
    "sh": STORE_TEMPLATE,
    "sw": STORE_TEMPLATE,

    "csrc": CSR_ALIAS_WRITE_TEMPLATE,
    "csrr": CSR_ALIAS_READ_TEMPLATE,
    "csrw": CSR_ALIAS_WRITE_TEMPLATE,
    "csrci": CSR_ALIAS_IMM_TEPLATE,
    "csrsi": CSR_ALIAS_IMM_TEPLATE,
    "csrwi": CSR_ALIAS_IMM_TEPLATE,
}

R64_INSTRUCTION_TEMPLATE = {
    # RV64
    "addiw": I_TEMPLATE,

    "slliw": I_TEMPLATE,
    "srliw": I_TEMPLATE,
    "sraiw": I_TEMPLATE,

    "addw": R_TEMPLATE,
    "subw": R_TEMPLATE,

    "sllw": R_TEMPLATE,
    "srlw": R_TEMPLATE,
    "sraw": R_TEMPLATE,

    "ld": LOAD_TEMPLATE,
    "lwu": LOAD_TEMPLATE,
    "sd": STORE_TEMPLATE
}

PSEUDO_INSTRUCTION_TEMPLATE = {
    "li": RI_TEMPLATE,
    "mv": R2_TEMPLATE,
    "not": R2_TEMPLATE,
    "neg": R2_TEMPLATE,
    "negw": R2_TEMPLATE,
    "sext.w": R2_TEMPLATE,
    "seqz": R2_TEMPLATE,
    "snez": R2_TEMPLATE,
    "sltz": R2_TEMPLATE,
    "sgtz": R2_TEMPLATE,
    "fmv.s": R2_TEMPLATE,
    "fabs.s": R2_TEMPLATE,
    "fneg.s": R2_TEMPLATE,
    "fmv.d": R2_TEMPLATE,
    "fabs.d": R2_TEMPLATE,
    "fneg.d": R2_TEMPLATE,

    "beqz": PSEUDO_B_TEMPLATE,
    "bnez": PSEUDO_B_TEMPLATE,
    "blez": PSEUDO_B_TEMPLATE,
    "bgez": PSEUDO_B_TEMPLATE,
    "bltz": PSEUDO_B_TEMPLATE,
    "bgtz": PSEUDO_B_TEMPLATE,
    "bgt": B_TEMPLATE,
    "ble": B_TEMPLATE,
    "bgtu": B_TEMPLATE,
    "bleu": B_TEMPLATE,
    "ret": NOP_TEMPLATE,
    "j": PSEUDO_J_TEMPLATE
}

CSR_INSTURCTION_TEMPLATE = {
    # Zicsr
    "csrrw": CSR_REG_TEMPLATE,
    "csrrs": CSR_REG_TEMPLATE,
    "csrrc": CSR_REG_TEMPLATE,
    "csrrwi": CSR_IMM_TEMPLATE,
    "csrrsi": CSR_IMM_TEMPLATE,
    "csrrci": CSR_IMM_TEMPLATE
}

MUL_INSTRUCTION_TEMPLATE = {
    # RV32M/64M
    "mul": R_TEMPLATE,
    "mulh": R_TEMPLATE,
    "mulhsu": R_TEMPLATE,
    "mulhu": R_TEMPLATE,
    "div": R_TEMPLATE,
    "divu": R_TEMPLATE,
    "rem": R_TEMPLATE,
    "remu": R_TEMPLATE,

    "mulw": R_TEMPLATE,
    "divw": R_TEMPLATE,
    "divuw": R_TEMPLATE,
    "remw": R_TEMPLATE,
    "remuw": R_TEMPLATE,
}

ATOMIC_INSTRUCTION_TEMPLATE = {
    "lr.w": ATOMIC_LOAD_TEMPLATE,
    "sc.w": ATOMIC_TEMPLATE,
    "amoadd.w": ATOMIC_TEMPLATE,
    "amoswap.w": ATOMIC_TEMPLATE,
    "amoxor.w": ATOMIC_TEMPLATE,
    "amoor.w": ATOMIC_TEMPLATE,
    "amoand.w": ATOMIC_TEMPLATE,
    "amomin.w": ATOMIC_TEMPLATE,
    "amomax.w": ATOMIC_TEMPLATE,
    "amominu.w": ATOMIC_TEMPLATE,
    "amomaxu.w": ATOMIC_TEMPLATE,

    "lr.d": ATOMIC_LOAD_TEMPLATE,
    "sc.d": ATOMIC_TEMPLATE,
    "amoadd.d": ATOMIC_TEMPLATE,
    "amoswap.d": ATOMIC_TEMPLATE,
    "amoxor.d": ATOMIC_TEMPLATE,
    "amoor.d": ATOMIC_TEMPLATE,
    "amoand.d": ATOMIC_TEMPLATE,
    "amomin.d": ATOMIC_TEMPLATE,
    "amomax.d": ATOMIC_TEMPLATE,
    "amominu.d": ATOMIC_TEMPLATE,
    "amomaxu.d": ATOMIC_TEMPLATE,
}

FLOATING_INSTRUCTION_TEMPLATE = {
    # RV32F/64F
    "flw": LOAD_TEMPLATE,
    "fsw": STORE_TEMPLATE,

    "fmadd.s": R4_TEMPLATE,
    "fmsub.s": R4_TEMPLATE,
    "fnmsub.s": R4_TEMPLATE,
    "fnmadd.s": R4_TEMPLATE,
    "fadd.s": R_TEMPLATE,
    "fsub.s": R_TEMPLATE,
    "fmul.s": R_TEMPLATE,
    "fdiv.s": R_TEMPLATE,
    "fsqrt.s": R2_TEMPLATE,
    "fmin.s" : R_TEMPLATE,
    "fmax.s" : R_TEMPLATE,

    "fcvt.w.s": R2_TEMPLATE,
    "fcvt.wu.s": R2_TEMPLATE,
    "fcvt.s.w": R2_TEMPLATE,
    "fcvt.s.wu": R2_TEMPLATE,

    "fsgnj.s": R_TEMPLATE,
    "fsgnjn.s": R_TEMPLATE,
    "fsgnjx.s": R_TEMPLATE,
    "fmv.x.w": R2_TEMPLATE,
    "fmv.w.x": R2_TEMPLATE,

    "feq.s": R_TEMPLATE,
    "flt.s": R_TEMPLATE,
    "fle.s": R_TEMPLATE,
    "fclass.s": R2_TEMPLATE,

    "fcvt.l.s": R2_TEMPLATE,
    "fcvt.lu.s": R2_TEMPLATE,
    "fcvt.s.l": R2_TEMPLATE,
    "fcvt.s.lu": R2_TEMPLATE,

    "fld": LOAD_TEMPLATE,
    "fsd": STORE_TEMPLATE,

    "fmadd.d": R4_TEMPLATE,
    "fmsub.d": R4_TEMPLATE,
    "fnmsub.d": R4_TEMPLATE,
    "fnmadd.d": R4_TEMPLATE,
    "fadd.d": R_TEMPLATE,
    "fsub.d": R_TEMPLATE,
    "fmul.d": R_TEMPLATE,
    "fdiv.d": R_TEMPLATE,
    "fsqrt.d": R2_TEMPLATE,
    "fmin.d" : R_TEMPLATE,
    "fmax.d" : R_TEMPLATE,

    "fcvt.d.s": R2_TEMPLATE,
    "fcvt.s.d": R2_TEMPLATE,
    "fcvt.w.d": R2_TEMPLATE,
    "fcvt.wu.d": R2_TEMPLATE,
    "fcvt.d.w": R2_TEMPLATE,
    "fcvt.d.wu": R2_TEMPLATE,

    "fsgnj.d": R_TEMPLATE,
    "fsgnjn": R_TEMPLATE,
    "fsgnjx": R_TEMPLATE,

    "feq.d": R_TEMPLATE,
    "flt.d": R_TEMPLATE,
    "fle.d": R_TEMPLATE,
    "fclass.d": R2_TEMPLATE,

    "fcvt.l.d": R2_TEMPLATE,
    "fcvt.lu.d": R2_TEMPLATE,
    "fcvt.d.l": R2_TEMPLATE,
    "fcvt.d.lu": R2_TEMPLATE,

    "fmv.x.d": R2_TEMPLATE,
    "fmv.d.x": R2_TEMPLATE
}

VECTOR_INSTRUCTION_TEMPLATE = {
    # Vector mask is not supported
    "vle8.v": VECTOR_LOAD_TEMPLATE,
    "vle16.v": VECTOR_LOAD_TEMPLATE,
    "vle32.v": VECTOR_LOAD_TEMPLATE,
    "vle64.v": VECTOR_LOAD_TEMPLATE,
    "vlm.v": VECTOR_LOAD_TEMPLATE,
    "vle8ff.v": VECTOR_LOAD_TEMPLATE,
    "vle16ff.v": VECTOR_LOAD_TEMPLATE,
    "vle32ff.v": VECTOR_LOAD_TEMPLATE,
    "vle64ff.v": VECTOR_LOAD_TEMPLATE,
    "vse8.v": VECTOR_STORE_TEMPLATE,
    "vse16.v": VECTOR_STORE_TEMPLATE,
    "vse32.v": VECTOR_STORE_TEMPLATE,
    "vse64.v": VECTOR_STORE_TEMPLATE,
    "vsm.v": VECTOR_STORE_TEMPLATE,
    "vse8ff.v": VECTOR_STORE_TEMPLATE,
    "vse16ff.v": VECTOR_STORE_TEMPLATE,
    "vse32ff.v": VECTOR_STORE_TEMPLATE,
    "vse64ff.v": VECTOR_STORE_TEMPLATE,

    "vlse8.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vlse16.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vlse32.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vlse64.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vsse8.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsse16.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsse32.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsse64.v": VECTOR_STRIDE_STORE_TEMPLATE,

    "vluxei8.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vluxei16.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vluxei32.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vluxei64.v": VECTOR_STRIDE_LOAD_TEMPLATE,

    "vloxei8.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vloxei16.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vloxei32.v": VECTOR_STRIDE_LOAD_TEMPLATE,
    "vloxei64.v": VECTOR_STRIDE_LOAD_TEMPLATE,

    "vsuxei8.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsuxei16.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsuxei32.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsuxei64.v": VECTOR_STRIDE_STORE_TEMPLATE,

    "vsoxei8.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsoxei16.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsoxei32.v": VECTOR_STRIDE_STORE_TEMPLATE,
    "vsoxei64.v": VECTOR_STRIDE_STORE_TEMPLATE,

    "vl1re8.v": VECTOR_LOAD_TEMPLATE,
    "vl1re16.v": VECTOR_LOAD_TEMPLATE,
    "vl1re32.v": VECTOR_LOAD_TEMPLATE,
    "vl1re64.v": VECTOR_LOAD_TEMPLATE,

    "vl2re8.v": VECTOR_LOAD_TEMPLATE,
    "vl2re16.v": VECTOR_LOAD_TEMPLATE,
    "vl2re32.v": VECTOR_LOAD_TEMPLATE,
    "vl2re64.v": VECTOR_LOAD_TEMPLATE,

    "vl4re8.v": VECTOR_LOAD_TEMPLATE,
    "vl4re16.v": VECTOR_LOAD_TEMPLATE,
    "vl4re32.v": VECTOR_LOAD_TEMPLATE,
    "vl4re64.v": VECTOR_LOAD_TEMPLATE,

    "vl8re8.v": VECTOR_LOAD_TEMPLATE,
    "vl8re16.v": VECTOR_LOAD_TEMPLATE,
    "vl8re32.v": VECTOR_LOAD_TEMPLATE,
    "vl8re64.v": VECTOR_LOAD_TEMPLATE,

    "vs1re8.v": VECTOR_STORE_TEMPLATE,
    "vs1re16.v": VECTOR_STORE_TEMPLATE,
    "vs1re32.v": VECTOR_STORE_TEMPLATE,
    "vs1re64.v": VECTOR_STORE_TEMPLATE,

    "vs2re8.v": VECTOR_STORE_TEMPLATE,
    "vs2re16.v": VECTOR_STORE_TEMPLATE,
    "vs2re32.v": VECTOR_STORE_TEMPLATE,
    "vs2re64.v": VECTOR_STORE_TEMPLATE,

    "vs4re8.v": VECTOR_STORE_TEMPLATE,
    "vs4re16.v": VECTOR_STORE_TEMPLATE,
    "vs4re32.v": VECTOR_STORE_TEMPLATE,
    "vs4re64.v": VECTOR_STORE_TEMPLATE,

    "vs8re8.v": VECTOR_STORE_TEMPLATE,
    "vs8re16.v": VECTOR_STORE_TEMPLATE,
    "vs8re32.v": VECTOR_STORE_TEMPLATE,
    "vs8re64.v": VECTOR_STORE_TEMPLATE,

    "vl1r.v": VECTOR_LOAD_TEMPLATE,
    "vl2r.v": VECTOR_LOAD_TEMPLATE,
    "vl4r.v": VECTOR_LOAD_TEMPLATE,
    "vl8r.v": VECTOR_LOAD_TEMPLATE,

    "vs1r.v": VECTOR_STORE_TEMPLATE,
    "vs2r.v": VECTOR_STORE_TEMPLATE,
    "vs4r.v": VECTOR_STORE_TEMPLATE,
    "vs8r.v": VECTOR_STORE_TEMPLATE,

    # For arithmetic vector instuction
    ".vv": VECTOR_VV_TEMPLATE,
    ".vx": VECTOR_VX_TEMPLATE,
    ".vi": VECTOR_VI_TEMPLATE,
    ".vf": VECTOR_VF_TEMPLATE,
    ".vv": VECTOR_VV_TEMPLATE,

    ".wv": VECTOR_WV_TEMPLATE,
    ".wx": VECTOR_WX_TEMPLATE
}

SUPPORTED_INSTRUCTION = [
    R32_INSTUCTION_TEMPLATE,
    R64_INSTRUCTION_TEMPLATE,
    PSEUDO_INSTRUCTION_TEMPLATE,
    CSR_INSTURCTION_TEMPLATE,
    MUL_INSTRUCTION_TEMPLATE,
    ATOMIC_INSTRUCTION_TEMPLATE,
    FLOATING_INSTRUCTION_TEMPLATE,
    VECTOR_INSTRUCTION_TEMPLATE
]

BRANCHES = ["beq", "bne", "blt", "bge", "bltu", "bgeu",
            "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
            "bgt", "ble", "bgtu", "bleu",
            "j",]

UNCONDITIONAL_JUMP = ["j", "ret"]

DRAM_LOAD = ["vl2re32.v"]
DRAM_STORE = ["vs2r.v"]

class rv_operand:
    def __init__(self, op_type, value) -> None:
        self.type = op_type & TYPE_MASK
        self.dest = op_type & DEST
        self.value = value
        self.offset = 0

        if op_type & MEM:
            self.value = value[1:-1]
        elif op_type & OFFSET:
            pos = value.find("(")
            self.offset = int(value[0:pos])
            self.value = value[pos+1:-1]
        else:
            self.value = value

    def is_imm(self):
        return self.type == IMMEDIATE

    def is_reg(self):
        return self.type == REGISTER

    def is_source(self):
        return not self.dest

    def is_destination(self):
        return self.dest

    def type_to_str(self):
        if self.type == REGISTER:
            return "register"
        elif self.type == SPECIAL:
            return "special"
        elif self.type == IMMEDIATE:
            return "immediate"
        elif self.type == LABEL:
            return "label"
        return "undefined"

    def __str__(self) -> str:
        value = f"Type:{self.type_to_str()}, Value:{self.value}"
        return value

class rv_instruction:
    def __init__(self, label, assembly_code:str):
        self.label = label
        self.asm = assembly_code.strip().rstrip()
        target_list = self.split_assembly(assembly_code)
        self.opcode = target_list[0]
        self.operands= []
        self.basic_block = None
        self.user_insts = []
        self.src_insts = []

        for inst_list in SUPPORTED_INSTRUCTION:
            if self.opcode not in inst_list:
                continue

            if len(inst_list[self.opcode]) != len(target_list[1:]):
                print(f"[Warn] {self.opcode}'s template mismatch in '{assembly_code}'")

            for op_type, value in zip(inst_list[self.opcode], target_list[1:]):
                self.operands.append(rv_operand(op_type, value))

            return

        # For vector extension code
        if self.opcode[0] == "v":
            for category, template in VECTOR_INSTRUCTION_TEMPLATE.items():
                if category not in self.opcode:
                    continue

                if len(template) != len(target_list[1:]):
                    print(f"[Warn] {self.opcode}'s template mismatch in '{assembly_code}'")

                for op_type, value in zip(template, target_list[1:]):
                    self.operands.append(rv_operand(op_type, value))

                return
        print(f"[Warn] Unsupported instruction in '{assembly_code.strip().rstrip()}'")

    def connect_user_inst(self, user_inst):
        self.user_insts.append(user_inst)
        user_inst.src_insts.append(self)

    def clear_dependency(self):
        self.user_insts = []
        self.src_insts = []

    @classmethod
    def split_assembly(cls, assembly_code:str):
        target_list = assembly_code.strip().replace(",", " ")
        comment_pos = target_list.find("#")
        if comment_pos != -1:
           target_list = target_list[0: comment_pos]

        target_list = target_list.split()
        return target_list

    @classmethod
    def is_label(cls, assembly_code:str):
        target_list = cls.split_assembly(assembly_code)

        if len(target_list) == 1:
            if target_list[0][-1] == ":":
                return True
        return False

    def __str__(self) -> str:
        join_str = "\n\t"
        operand_str = [str(op) for op in self.operands]
        info =  f"[OPCODE]:{self.opcode}"
        if self.label != "":
            info = f"{info}, Label: {self.label}"
        info = f"{info}\n\t"
        return f"{info}{join_str.join(operand_str)}"


class loop:
    def __init__(self, path) -> None:
        self.loop_path = OrderedDict()
        for idx, bb in enumerate(path):
            self.loop_path[idx] = bb

    def __equal__(self, path):
        return set(self.loop_path.values()) == set(path)

    def __str__(self) -> str:
        join_str = "->"
        return join_str.join([str(bb.name) for bb in self.loop_path.values()])

    def iter_insts(self):
        return chain.from_iterable([iter(bb) for bb in self.loop_path.values()])


NR_BLOCK = 0
class basic_block:
    def __init__(self, name=""):
        self.inputs = []
        self.outputs = []
        self.visited = False
        self.cycle_list = []

        if name != "":
            self.name = f"{name}"
        else:
            global NR_BLOCK
            self.name = f"BasicBlock{NR_BLOCK}"
            NR_BLOCK += 1
        self.insts = []

    def connect(self, new_block):
        self.outputs.append(new_block)
        new_block.inputs.append(self)

    def add_inst(self, inst):
        self.insts.append(inst)
        inst.basic_block = self

    def to_onnx(self):
        inputs = [i.name + "_output" for i in self.inputs]
        outputs = [self.name + "_output"]
        #asm = "\n"
        lines = {}
        asm = ([i.asm.strip().rstrip() for i in self.insts])
        for idx, asm_line in enumerate(asm):
            lines[f"inst{idx}"] = asm_line

        onnx_node = onnx.helper.make_node(op_type=self.__class__.__name__,
                                          inputs=inputs,
                                          outputs=outputs,
                                          bb_name = self.name,
                                          **lines)
        return onnx_node

    def dfs(self, start, path=[]):
        if self.visited:
            if self == start:
                tmp_loop = loop(path)
                for bb in path:
                    is_duplicated = any([tmp_loop == cycle for cycle in bb.cycle_list])
                    if not is_duplicated:
                        bb.cycle_list.append(tmp_loop)
            return

        self.visited = True
        path.append(self)
        for child in self.outputs:
            child.dfs(start, path)
        path.pop(-1)
        self.visited = False

    def __iter__(self):
        return iter(self.insts)

class riscv_parser:
    def __init__(self) -> None:
        self.inst_list = []
        self.bb_list = []
        self.cycle_list = []

    def load_file(self, name):
        f = open("vectoradd.s")
        asm_lines = f.readlines()[1:]

        label = ""
        for asm_line in asm_lines:
            if rv_instruction.is_label(asm_line):
                label = rv_instruction.split_assembly(asm_line)[0][:-1]
                continue
            self.inst_list.append(rv_instruction(label, asm_line))
            label = ""

    def basic_block_analysis(self):
        # Construct Basic Block
        bb = basic_block()
        self.bb_list.append(bb)
        for inst in self.inst_list:
            if inst.label != "":
                bb = basic_block(inst.label)
                self.bb_list.append(bb)
            bb.add_inst(inst)

            if inst.opcode in BRANCHES:
                bb = basic_block()
                self.bb_list.append(bb)

        # Trim empty Basic Block
        self.bb_list = [bb for bb in self.bb_list if len(bb.insts)]

        # Link Basic Block
        prev_inst = self.inst_list[0]
        for inst in self.inst_list[1:]:
            if prev_inst.basic_block != inst.basic_block and prev_inst.opcode not in UNCONDITIONAL_JUMP:
                prev_inst.basic_block.connect(inst.basic_block)

            if inst.opcode in BRANCHES:
                labels = [op.value for op in inst.operands if op.type & LABEL]
                for label in labels:
                    for iter_bb in self.bb_list:
                        if iter_bb.name != label:
                            continue
                        inst.basic_block.connect(iter_bb)

            # Update prev inst
            prev_inst = inst

    def dump_basic_block_graph(self, name):
        # Dump to onnx model
        onnx_node_list = []
        for bb in self.bb_list:
            onnx_node_list.append(bb.to_onnx())

        graph_def = onnx.helper.make_graph(
            inputs=[],#load_tile_name1, load_tile_name2],
            outputs=[],#store_tile_name],
            nodes=onnx_node_list,
            name="Dummy tile graph",
        )
        model_def = onnx.helper.make_model(graph_def, producer_name="PyTorchSim")
        model_def.opset_import[0].version = 13

        onnx.save(model_def, name)

    def cycle_detect_analysis(self):
        for bb in self.bb_list:
            bb.dfs(bb, [])
            for bb_cycle in bb.cycle_list:
                is_duplicated = any([bb_cycle == cycle for cycle in self.cycle_list])
                if not is_duplicated:
                    self.cycle_list.append(bb_cycle)

    def print_cycles(self):
        for cycle in self.cycle_list:
            print(f"Cycle-path: {cycle}")

    def cycle_analysis(self, *args, **kwargs):
        for idx, cycle in enumerate(self.cycle_list):
            # Construct rough instruction dependency
            scoreboard = {}
            for inst in cycle.iter_insts():
                for op in inst.operands[::-1]:
                    if op.is_destination() and op.is_reg():
                        scoreboard[op.value] = inst
                    elif op.is_reg() and op.value in scoreboard:
                        scoreboard[op.value].connect_user_inst(inst)

            # Cycle analysis phase start
            self.generate_tile_graph(cycle, *args, suffix=idx, **kwargs)

            # Clear instruction dependency
            for inst in cycle.iter_insts():
                inst.clear_dependency()

    def generate_tile_graph(self, cycle, name="tile_graph", suffix=0):
        current_pointer = {}
        for inst in cycle.iter_insts():
            if inst.opcode in DRAM_LOAD:
                tmp_node = load_node([inst.asm], len(current_pointer))
                current_pointer[inst] = tmp_node
            elif inst.opcode in DRAM_STORE:
                tmp_node = store_node([inst.asm], len(current_pointer))
                neighbor_node = [current_pointer[user_inst] for user_inst in inst.src_insts if user_inst in current_pointer]
                check_compute = [node for node in neighbor_node if isinstance(node, compute_node)]
                for source_node in check_compute:
                        connect_nodes(source_node, tmp_node)
                current_pointer[inst] = tmp_node
            else:
                neighbor_node = [current_pointer[user_inst] for user_inst in inst.src_insts if user_inst in current_pointer]
                check_compute = [node for node in neighbor_node if isinstance(node, compute_node)]
                check_load = [node for node in neighbor_node if isinstance(node, load_node)]

                if len(check_compute):
                    check_compute[0].inst.append(inst.asm)

                elif len(check_load):
                    tmp_node = compute_node([inst.asm], len(current_pointer))
                    for source_node in check_load:
                        connect_nodes(source_node, tmp_node)
                    current_pointer[inst] = tmp_node

        onnx_node_list = [node.to_onnx() for node in current_pointer.values()]
        if onnx_node_list:
            dump_onnx_graph(f"{name}_{suffix}.onnx", onnx_node_list)

if __name__ == "__main__":
    # For Test!
    parser = riscv_parser()
    parser.load_file("vectoradd.s")

    parser.basic_block_analysis()
    parser.dump_basic_block_graph("basic_block.onnx")
    parser.cycle_detect_analysis()
    parser.print_cycles()
    parser.cycle_analysis()