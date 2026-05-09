; fibonacci.asm — print Fibonacci numbers that fit in 8 bits (0–233)
; $80 = a (previous), $81 = b (current), $82 = temp
.org $00
    LDA #0
    STA $80       ; a = 0
    LDA #1
    STA $81       ; b = 1

loop:
    LDA $80
    OUTN          ; print a
    LDA #$20      ; space
    OUT

    ; temp = a + b
    LDA $80
    ADD $81
    STA $82       ; temp = a + b

    ; if carry or result < a (overflow into >255), stop
    JC  done

    ; a = b
    LDA $81
    STA $80

    ; b = temp
    LDA $82
    STA $81

    JMP loop

done:
    LDA #$0A
    OUT
    HLT
