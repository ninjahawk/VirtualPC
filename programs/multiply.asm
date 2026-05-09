; multiply.asm — read two numbers from the user, print the product.
; Uses the gate-level MUL opcode (mul8 in alu.py = shift-and-add via NAND).
; Result wraps mod 256 since registers are 8-bit.
.org $00
    LDA #$66      ; 'f'
    OUT
    LDA #$69      ; 'i'
    OUT
    LDA #$72      ; 'r'
    OUT
    LDA #$73      ; 's'
    OUT
    LDA #$74      ; 't'
    OUT
    LDA #$3A      ; ':'
    OUT
    LDA #$20      ; ' '
    OUT
    INP
    STA $80       ; a = first

    LDA #$73      ; 's'
    OUT
    LDA #$65      ; 'e'
    OUT
    LDA #$63      ; 'c'
    OUT
    LDA #$6F      ; 'o'
    OUT
    LDA #$6E      ; 'n'
    OUT
    LDA #$64      ; 'd'
    OUT
    LDA #$3A      ; ':'
    OUT
    LDA #$20      ; ' '
    OUT
    INP           ; A = second

    MUL $80       ; A = A * a  (gate-level multiplier)
    STA $81

    LDA #$3D      ; '='
    OUT
    LDA #$20      ; ' '
    OUT
    LDA $81
    OUTN          ; print product
    LDA #$0A
    OUT
    HLT
