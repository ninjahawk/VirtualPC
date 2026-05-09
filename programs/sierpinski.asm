; sierpinski.asm — Pascal's triangle mod 2, drawn live.
; Each row is computed from the previous via new[i] = old[i-1] XOR old[i].
; The XOR runs through the gate-level NAND chain in alu.py — every '#' you
; see was decided by a NAND gate.
;
; Buffer: $00..$3F (64 cells, 1 byte each, value 0 or 1).
; $40 = prev (value of cell to the left, last row).
; $41 = row counter.
; $42 = saved 'cur' during the in-place update.
.org $00
    ; zero the row buffer
    LDX #0
zloop:
    LDA #0
    STAX
    TXA
    ADD #1
    TAX
    CMP #64
    JNZ zloop

    ; seed: a single 1 in the middle (cell 32)
    LDX #32
    LDA #1
    STAX

    LDA #0
    STA $41       ; row = 0

print_row:
    LDX #0
prloop:
    LDAX
    JZ  pspc
    LDA #$23      ; '#'
    OUT
    JMP padv
pspc:
    LDA #$20      ; ' '
    OUT
padv:
    TXA
    ADD #1
    TAX
    CMP #64
    JNZ prloop
    LDA #$0A
    OUT

    ; stop after 32 rows
    LDA $41
    ADD #1
    STA $41
    CMP #32
    JZ  done

    ; compute next row in place: new[i] = prev XOR cur; prev := cur
    LDA #0
    STA $40
    LDX #0
crloop:
    LDAX
    STA $42       ; save cur
    XOR $40       ; A = prev XOR cur  (gate-level XOR via NAND)
    STAX
    LDA $42
    STA $40       ; prev = cur

    TXA
    ADD #1
    TAX
    CMP #64
    JNZ crloop

    JMP print_row

done:
    HLT
