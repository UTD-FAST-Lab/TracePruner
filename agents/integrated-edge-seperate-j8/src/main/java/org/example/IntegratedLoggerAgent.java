package org.example;

import java.lang.instrument.Instrumentation;

public class IntegratedLoggerAgent {

    enum AgentLevel{
        BRANCH,
        CG,
        VAR,
        FULL
    }

    enum LogLevel{
        PACKAGE,
        CLASS,
        METHOD
    }


    public static void premain(String agentArgs, Instrumentation inst) {
        System.out.println("starting the agent");
        AgentLevel myAgent = AgentLevel.FULL;
        LogLevel myLog = LogLevel.METHOD;

        if (agentArgs != null){
            String[] options = agentArgs.split(",");
            for (String option : options) {
                if (option.startsWith("agentLevel=")) {
                    String agentlevel = option.split("=")[1];
                    System.out.println("Agent level set to: " + agentlevel);

                    switch (agentlevel){
                        case "branch":
                            myAgent = AgentLevel.BRANCH;
                            break;
                        case "cg":
                            myAgent = AgentLevel.CG;
                            break;
                        case "var":
                            myAgent = AgentLevel.VAR;
                            break;
                        default:
                            myAgent = AgentLevel.FULL;
                            break;
                    }
                } 

                else if (option.startsWith("logLevel=")) {
                    String loglevel = option.split("=")[1];
                    System.out.println("Log level set to: " + loglevel);

                    switch (loglevel){
                        case "package":
                            myLog = LogLevel.PACKAGE;
                            break;
                        case "class":
                            myLog = LogLevel.CLASS;
                            break;
                        case "method":
                            myLog = LogLevel.METHOD;
                            break;
                        default:
                            myLog = LogLevel.METHOD;
                            break;
                    }
                }
            }
        }

     


        inst.addTransformer(new IntegratedLoggerTransformer(myAgent, myLog));
    }
}


