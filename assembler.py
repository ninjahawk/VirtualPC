# assembler.py — Two-pass assembler for VirtualPC assembly language
#
# Syntax:
#   label:          define a label
#   LDA #$FF        immediate (hex)
#   LDA #255        immediate (decimal)
#   LDA #%11111111  immediate (binary)
#   LDA $80         memory address
#   JMP label       jump to label
#   .org $00        set origin address
#   .byte $01,$02   raw bytes
#   .str "hello\n"  string bytes
#   ; comment

import re
from cpu import (
    NOP, LDA_IMM, LDA_ADDR, STA_ADDR,
    DRAW_OP, KEY_OP, WAIT_OP, MUL_IMM, MUL_ADDR,
    JMPL, JZL, JNZL, JCL, JNCL, JNL,
    ADD_IMM, ADD_ADDR, SUB_IMM, SUB_ADDR,
    AND_IMM, AND_ADDR, OR_IMM, OR_ADDR,
    XOR_IMM, XOR_ADDR, NOT_OP,
    JMP_ADDR, JZ_ADDR, JNZ_ADDR, JC_ADDR, JNC_ADDR, JN_ADDR,
    INC_OP, DEC_OP, OUT_OP, OUTA_OP, OUTN_OP, INP_OP,
    LDX_IMM, LDX_ADDR, STX_ADDR, TAX_OP, TXA_OP,
    CMP_IMM, CMP_ADDR,
    PUSH_OP, POP_OP, JSR_ADDR, RET_OP,
    LDAX_OP, STAX_OP, SHL_OP, SHR_OP, HLT,
)

# Mnemonics that take no operand
NO_OPERAND = {
    'NOP': NOP, 'NOT': NOT_OP, 'INC': INC_OP, 'DEC': DEC_OP,
    'OUT': OUT_OP, 'OUTA': OUTA_OP, 'OUTN': OUTN_OP, 'INP': INP_OP,
    'TAX': TAX_OP, 'TXA': TXA_OP,
    'PUSH': PUSH_OP, 'POP': POP_OP, 'RET': RET_OP,
    'LDAX': LDAX_OP, 'STAX': STAX_OP,
    'SHL': SHL_OP, 'SHR': SHR_OP,
    'DRAW': DRAW_OP, 'KEY': KEY_OP, 'WAIT': WAIT_OP,
    'HLT': HLT,
}

# Mnemonics with one operand that is always an address (not immediate)
ADDR_ONLY = {
    'STA': STA_ADDR, 'JMP': JMP_ADDR,
    'JZ': JZ_ADDR, 'JNZ': JNZ_ADDR,
    'JC': JC_ADDR, 'JNC': JNC_ADDR, 'JN': JN_ADDR,
    'STX': STX_ADDR, 'JSR': JSR_ADDR,
}

# Mnemonics with two variants: immediate (#) or address
IMM_ADDR = {
    'LDA': (LDA_IMM, LDA_ADDR),
    'ADD': (ADD_IMM, ADD_ADDR),
    'SUB': (SUB_IMM, SUB_ADDR),
    'AND': (AND_IMM, AND_ADDR),
    'OR':  (OR_IMM,  OR_ADDR),
    'XOR': (XOR_IMM, XOR_ADDR),
    'LDX': (LDX_IMM, LDX_ADDR),
    'CMP': (CMP_IMM, CMP_ADDR),
    'MUL': (MUL_IMM, MUL_ADDR),
}

# Long (16-bit address) jump variants — 3 bytes each
JUMP16 = {
    'JMPL': JMPL, 'JZL': JZL, 'JNZL': JNZL,
    'JCL': JCL, 'JNCL': JNCL, 'JNL': JNL,
}

def parse_num(s):
    s = s.strip()
    if s.startswith('$'):    return int(s[1:], 16)
    if s.startswith('0x'):   return int(s, 16)
    if s.startswith('%'):    return int(s[1:], 2)
    return int(s)

def _line_size(mnemonic, operand):
    if mnemonic in NO_OPERAND:
        return 1
    if mnemonic in JUMP16:
        return 3
    if mnemonic in ADDR_ONLY or mnemonic in IMM_ADDR:
        return 2
    return 0

def assemble(source):
    """Return (bytecode, origin, labels_dict, errors_list)"""
    errors = []
    labels = {}
    origin = 0

    # ── Pass 1: collect labels ────────────────────────────────────────────────
    pc = origin
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('.org'):
            try: origin = pc = parse_num(line.split()[1])
            except Exception: errors.append(f"line {lineno}: bad .org")
            continue
        if line.startswith('.byte'):
            vals = [v.strip() for v in line[5:].split(',') if v.strip()]
            pc += len(vals)
            continue
        if line.startswith('.str'):
            m = re.search(r'"(.*?)"', line)
            if m:
                s = m.group(1).replace('\\n','\n').replace('\\t','\t').replace('\\r','\r')
                pc += len(s.encode())
            continue
        if ':' in line:
            lbl, _, rest = line.partition(':')
            lbl = lbl.strip()
            if lbl:
                labels[lbl] = pc
            line = rest.strip()
            if not line:
                continue
        parts = line.split()
        mn = parts[0].upper()
        op = parts[1] if len(parts) > 1 else None
        pc += _line_size(mn, op) or (2 if op else 1)

    # ── Pass 2: emit bytes ────────────────────────────────────────────────────
    output = {}   # addr -> byte
    pc = origin

    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('.org'):
            try: origin = pc = parse_num(line.split()[1])
            except Exception: pass
            continue
        if line.startswith('.byte'):
            for v in line[5:].split(','):
                v = v.strip()
                if v:
                    try: output[pc] = parse_num(v) & 0xFF; pc += 1
                    except Exception: errors.append(f"line {lineno}: bad byte '{v}'")
            continue
        if line.startswith('.str'):
            m = re.search(r'"(.*?)"', line)
            if m:
                s = m.group(1).replace('\\n','\n').replace('\\t','\t').replace('\\r','\r')
                for ch in s.encode():
                    output[pc] = ch; pc += 1
            continue
        if ':' in line:
            _, _, line = line.partition(':')
            line = line.strip()
            if not line:
                continue

        parts = line.split()
        if not parts:
            continue
        mn = parts[0].upper()
        operand_str = parts[1] if len(parts) > 1 else None

        # Resolve operand value
        val = None
        val16 = None   # for JUMP16 instructions, keep full 16-bit address
        is_imm = False
        if operand_str:
            if operand_str.startswith('#'):
                is_imm = True
                try: val = parse_num(operand_str[1:]) & 0xFF
                except Exception:
                    errors.append(f"line {lineno}: bad immediate '{operand_str}'"); continue
            else:
                try: raw_val = parse_num(operand_str)
                except ValueError:
                    if operand_str in labels:
                        raw_val = labels[operand_str]
                    else:
                        errors.append(f"line {lineno}: undefined label '{operand_str}'"); continue
                val16 = raw_val & 0xFFFF
                val   = raw_val & 0xFF

        # Emit opcode
        if mn in JUMP16:
            output[pc] = JUMP16[mn]; pc += 1
            if val16 is not None:
                output[pc] = (val16 >> 8) & 0xFF; pc += 1
                output[pc] = val16 & 0xFF;         pc += 1
            else:
                errors.append(f"line {lineno}: {mn} needs an address operand")
            continue
        if mn in NO_OPERAND:
            output[pc] = NO_OPERAND[mn]; pc += 1
        elif mn in ADDR_ONLY:
            output[pc] = ADDR_ONLY[mn]; pc += 1
            if val is not None: output[pc] = val; pc += 1
            else: errors.append(f"line {lineno}: {mn} needs an operand")
        elif mn in IMM_ADDR:
            imm_op, addr_op = IMM_ADDR[mn]
            output[pc] = imm_op if is_imm else addr_op; pc += 1
            if val is not None: output[pc] = val; pc += 1
            else: errors.append(f"line {lineno}: {mn} needs an operand")
        else:
            errors.append(f"line {lineno}: unknown mnemonic '{mn}'")

    # Pack into bytes starting at origin
    if not output:
        return bytes(), origin, labels, errors
    lo = min(output); hi = max(output)
    result = bytearray(hi - lo + 1)
    for addr, byte in output.items():
        result[addr - lo] = byte
    return bytes(result), lo, labels, errors
