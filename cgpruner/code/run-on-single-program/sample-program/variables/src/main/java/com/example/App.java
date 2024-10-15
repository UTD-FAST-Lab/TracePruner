package com.example;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        int a = 5;
        int b;

        if (a < 10){
            b = 10;
        }
        else{
            b = 12;
        }

        foo(b);
    }

    static void foo(int b){
        int c = b;
    } 
}
