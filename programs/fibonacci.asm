; fibonacci.asm — print the first N Fibonacci numbers (8-bit).
; Stops early if the next term overflows 255.
; $80 = a (prev)   $81 = b (curr)   $82 = remaining count   $83 = temp
.org $00
    LDA #$6E      ; 'n'
    OUT
    LDA #$3A      ; ':'
    OUT
    LDA #$20
    OUT
    INP
    STA $82       ; count = n

    LDA #0
    STA $80       ; a = 0
    LDA #1
    STA $81       ; b = 1

loop:
    LDA $82
    CMP #0
    JZ  done

    LDA $80
    OUTN          ; print a
    LDA #$20      ; space
    OUT

    LDA $80
    ADD $81
    JC  done      ; overflow: stop cleanly
    STA $83       ; temp = a + b

    LDA $81
    STA $80       ; a = b
    LDA $83
    STA $81       ; b = temp

    LDA $82
    SUB #1
    STA $82       ; count -= 1

    JMP loop

done:
    LDA #$0A
    OUT
    HLT
