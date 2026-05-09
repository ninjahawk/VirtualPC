; factorial.asm — read N, print N!  (8-bit; 6! = 720 already overflows).
; Strategy: result = 1; for i = N down to 2: result *= i
; $80 = result   $81 = i
.org $00
    LDA #$6E      ; 'n'
    OUT
    LDA #$3A
    OUT
    LDA #$20
    OUT
    INP
    STA $81       ; i = N

    LDA #1
    STA $80       ; result = 1

    LDA $81
    CMP #0
    JZ  done      ; 0! = 1, skip the loop

loop:
    LDA $81
    CMP #1
    JZ  done      ; i == 1: done (multiplying by 1 is a no-op)

    LDA $80
    MUL $81       ; result *= i
    STA $80

    LDA $81
    SUB #1
    STA $81       ; i -= 1
    JMP loop

done:
    LDA $80
    OUTN
    LDA #$0A
    OUT
    HLT
