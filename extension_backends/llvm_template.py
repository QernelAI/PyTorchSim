import torch
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.codegen.common import ChoiceCaller
from torch._inductor.codegen.common import Kernel
from torch._inductor.ir import TensorBox
from torch._inductor.ir import TemplateBuffer
from torch._inductor.utils import override_lowering

aten = torch.ops.aten

class LLVMTemplateKernel(Kernel):
    def __init__(self) -> None:
        pass

    def def_kernel(self, inputs, outputs, names_str):
        pass

    def call_kernel(self, name, node):
        pass

class LLVMTemplateCaller(ChoiceCaller):
    def call_name(self):
        pass

    def to_callable(self):
        pass

    def hash_key(self):
        pass

    def output_node(self) -> "TensorBox":
        return TensorBox.create(
            TemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                workspace_size=self.bmreq.workspace_size,
                template=self.template,
            )
        )

class LLVMTemplate(KernelTemplate):
    def maybe_append_choice(self, choices, **kwargs):
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """

        try:
            choices.append(self.generate(**kwargs))
        except NotImplementedError:
            pass

    def generate(self, **kwargs) -> ChoiceCaller:
        kernel_name = f"llvm_{self.name}"
        kernel  = LLVMTemplateKernel(kernel_name=kernel_name)
        code = self.render(kernel=kernel, **kwargs)
        _, call_args, _ = kernel.args.python_argdefs()

        return LLVMTemplateCaller(
            kernel_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            code,
            self,
        ).output_node() # Skip autotuning process at now!

class LLVMGemmTemplate(LLVMTemplate):
    def render(self):
        pass

@override_lowering(aten.mm)
def tuned_mm(mat1, mat2, * ,layout=None):
    llvm_template = LLVMGemmTemplate()

    return llvm_template.generate()