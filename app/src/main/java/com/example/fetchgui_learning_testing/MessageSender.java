package com.example.fetchgui_learning_testing;
import android.os.AsyncTask;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.InetAddress;

public class MessageSender extends AsyncTask<String,Void,Void>{

    Socket s;
    DataOutputStream dos;

    PrintWriter pw;

    @Override
    protected Void doInBackground(String... voids) {

        String message = voids[0];
        String ipAddress = "192.168.1.8";
        int port = 5000;
        try{
            s = new Socket(ipAddress, port);
            pw = new PrintWriter((s.getOutputStream()));
            pw.write(message);
            pw.flush();
            pw.close();
            s.close();
        }
        catch(IOException e){
            e.printStackTrace();
        }

        return null;
    }
}
