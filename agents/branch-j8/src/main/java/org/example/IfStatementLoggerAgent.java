package org.example;

import java.lang.instrument.Instrumentation;

public class IfStatementLoggerAgent {
    public static void premain(String agentArgs, Instrumentation inst) {
        System.out.println("starting the agent");
        inst.addTransformer(new IfStatementLoggerTransformer());
    }
}
