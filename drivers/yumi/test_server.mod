MODULE test_server

    VAR socketdev clientSocket;
    VAR socketdev serverSocket;
    VAR num instructionCode;
    VAR num params{250};
    VAR num nParams;
    PERS string ipController:="192.168.125.1";
    VAR num serverPort:=5001;
    PROC ServerCreateAndConnect(string ip,num port)
        VAR string clientIP;

        SocketCreate serverSocket;
        SocketBind serverSocket,ip,port;
        SocketListen serverSocket;

        !! ASG: while "current socket status of clientSocket" IS NOT EQUAL TO the "client connected to a remote host"
        WHILE SocketGetStatus(clientSocket)<>SOCKET_CONNECTED DO
            SocketAccept serverSocket,clientSocket\ClientAddress:=clientIP\Time:=WAIT_MAX;
            !//Wait 0.5 seconds for the next reconnection
            WaitTime 0.5;
        ENDWHILE
    ENDPROC

    PROC ReceiveMsg(\num wait_time)
        VAR rawbytes buffer;
        VAR num time_val := WAIT_MAX;  ! default to wait-forever
        VAR num bytes_rcvd;

        TPErase;
        TPWrite "START RECEIVING";

        ClearRawBytes buffer;
        IF Present(wait_time) time_val := wait_time;    ! test if wait time is setted
        SocketReceive clientSocket, \RawData:=buffer, \ReadNoOfBytes:=1024, \NoRecBytes:=bytes_rcvd, \Time:=time_val;
        UnpackRawBytes buffer, 1, nParams, \IntX:=INT;
        UnpackRawBytes buffer, 3, instructionCode, \IntX:=USINT;
        ! Read remaining message bytes
        TPWrite NumToStr(nParams,0);
        TPWrite NumToStr(instructionCode,0);
        TPWrite "No of bytes " + NumToStr(bytes_rcvd, 0);
        TPWrite "No of parameters " + NumToStr((bytes_rcvd-4)/4,5);
        ! parameters are start from 5
        IF (bytes_rcvd-4)/4 <> nParams THEN
            ! TODO Deal with error
            ErrWrite \W, "Socket Recv Failed", "Did not receive expected # of bytes.",
                 \RL2:="Expected: " + ValToStr(nParams),
                 \RL3:="Received: " + ValToStr((bytes_rcvd-3)/4);
        ELSE
            FOR i FROM 1 TO nParams DO
                UnpackRawBytes buffer, 5 + (i-1)*4, params{i}, \Float4;
            ENDFOR
        ENDIF
    ENDPROC

    PROC main()
        !//Local variables
        VAR string receivedString;
        !//String to add to the reply.
        VAR bool connected;
        !//Reconnect after sending ack
        VAR bool reconnect;
        !//Reply string
        connected:=FALSE;
        ServerCreateAndConnect ipController,serverPort;
        connected:=TRUE;
        reconnect:=FALSE;
        ReceiveMsg;
        !SocketReceive clientSocket\Str:=receivedString\Time:=WAIT_MAX;
        !TPWrite receivedString;
        Stop;

    ENDPROC


ENDMODULE