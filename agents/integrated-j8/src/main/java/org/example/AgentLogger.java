package org.example;


public class AgentLogger {
    public static void logCGEdge(String className, String methodName, String desc, String owner, String name, String descriptor) {
        String message = "AgentLogger|CG_edge: " + className.replace('/', '.') + "." + methodName + " " + desc +
                         " -> " + owner.replace('/', '.') + "." + name + " " + descriptor;
        System.out.println(message);
    }

    public static void logBranch(String className, String methodName, String desc, int ifCounter) {
        String ifStatementId = className.replace('/', '.') + "." + methodName + " " + desc + ":IF#" + ifCounter;
        System.out.println("AgentLogger|BRANCH: " + ifStatementId);
    }

    public static void logVariable(String className, String methodName, String desc, String varName, int varValue) {
        String message = "AgentLogger|VARIABLE: " + className.replace('/', '.') + "." + methodName + " " + desc + varName + " = " + varValue;
        System.out.println(message);
    }

    public static void logVariable(String className, String methodName, String desc, String varName, Object varValue) {
        String message = "AgentLogger|VARIABLE: " + className.replace('/', '.') + "." + methodName + " " + desc + varName + " = " + varValue;
        System.out.println(message);
    }
}