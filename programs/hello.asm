; hello.asm — print "Hello, World!" using OUT (ASCII)
.org $00
    LDA #$48  ; H
    OUT
    LDA #$65  ; e
    OUT
    LDA #$6C  ; l
    OUT
    OUT       ; l  (A still holds $6C)
    LDA #$6F  ; o
    OUT
    LDA #$2C  ; ,
    OUT
    LDA #$20  ; space
    OUT
    LDA #$57  ; W
    OUT
    LDA #$6F  ; o
    OUT
    LDA #$72  ; r
    OUT
    LDA #$6C  ; l
    OUT
    LDA #$64  ; d
    OUT
    LDA #$21  ; !
    OUT
    LDA #$0A  ; newline
    OUT
    HLT
