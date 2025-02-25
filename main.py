from argu import ABAF
import sys

framework_filename = "examples/ABA_ICCMA_input.txt"

aba = ABAF()
aba.create_from_file(framework_filename)

aba.print_ABA()