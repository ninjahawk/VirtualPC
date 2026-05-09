; multiply.asm — multiply two numbers via repeated addition
; 6 * 7 = 42
; $80 = multiplicand (6), $81 = multiplier (7, loop counter), $82 = result
.org $00
    LDA #6
    STA $80       ; a = 6
    LDA #7
    STA $81       ; b = 7 (loop counter)
    LDA #0
    STA $82       ; result = 0

loop:
    LDA $81
    CMP #0
    JZ  done      ; if counter == 0, done

    LDA $82
    ADD $80
    STA $82       ; result += a

    LDA $81
    SUB #1
    STA $81       ; counter--

    JMP loop

done:
    LDA #$36      ; '6'
    OUT
    LDA #$20
    OUT
    LDA #$2A      ; '*'
    OUT
    LDA #$20
    OUT
    LDA #$37      ; '7'
    OUT
    LDA #$20
    OUT
    LDA #$3D      ; '='
    OUT
    LDA #$20
    OUT
    LDA $82
    OUTN          ; print result
    LDA #$0A
    OUT
    HLT
