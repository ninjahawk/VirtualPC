; add.asm — read two numbers and print the sum (mod 256).
; Carry-out flag is reported on the next line for honesty.
.org $00
    LDA #$61      ; 'a'
    OUT
    LDA #$3A      ; ':'
    OUT
    LDA #$20
    OUT
    INP
    STA $80

    LDA #$62      ; 'b'
    OUT
    LDA #$3A
    OUT
    LDA #$20
    OUT
    INP

    ADD $80       ; A = b + a, carry flag set if it overflowed
    STA $81       ; sum
    JC  with_carry

    LDA #$3D      ; '='
    OUT
    LDA #$20
    OUT
    LDA $81
    OUTN
    LDA #$0A
    OUT
    HLT

with_carry:
    LDA #$3D
    OUT
    LDA #$20
    OUT
    LDA $81
    OUTN
    LDA #$20
    OUT
    LDA #$28      ; '('
    OUT
    LDA #$2B      ; '+'
    OUT
    LDA #$32      ; '2'
    OUT
    LDA #$35      ; '5'
    OUT
    LDA #$36      ; '6'
    OUT
    LDA #$29      ; ')'
    OUT
    LDA #$0A
    OUT
    HLT
