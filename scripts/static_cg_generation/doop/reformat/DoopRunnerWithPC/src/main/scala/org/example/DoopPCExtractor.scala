package org.example

import java.io.File
import java.io.PrintWriter
import java.io.Writer
import java.net.URL
import java.nio.file.Files

import scala.collection.compat.immutable
import scala.collection.mutable
import scala.io.Source
import scala.sys.process.Process
import scala.util.matching.Regex

import org.apache.commons.io.FileUtils
import play.api.libs.json.Json
import play.api.libs.json.JsValue

import org.opalj.br.ClassFile
import org.opalj.br.FieldType
import org.opalj.br.FieldTypes
import org.opalj.br.MethodDescriptor
import org.opalj.br.ObjectType
import org.opalj.br.ReferenceType
import org.opalj.br.ReturnType
import org.opalj.br.analyses.Project
import org.opalj.br.analyses.SomeProject
import org.opalj.br.instructions.Instruction
import org.opalj.br.instructions.INVOKEDYNAMIC
import org.opalj.br.instructions.MethodInvocationInstruction

object DoopPCExtractor {

  private def resolveBridgeMethod(
      classFile: ClassFile,
      p: SomeProject,
      bridgeMethod: org.opalj.br.Method
  ): org.opalj.br.Method = {
    val methods = classFile.findMethod(bridgeMethod.name).filter { m =>
      !m.isBridge && (m.returnType match {
        case rt: ReferenceType => p.classHierarchy.isSubtypeOf(rt, bridgeMethod.returnType.asReferenceType)
        case rt                => rt == bridgeMethod.returnType
      })
    }
    assert(methods.size == 1)
    methods.head
  }

  private def computePC(
      classFile: ClassFile,
      p: SomeProject,
      declaredTgt: String,
      number: Int,
      method: org.opalj.br.Method
  ): Option[Int] = {
    val split = declaredTgt.split("""\.""")
    val declaredType = s"L${split.slice(0, split.size - 1).mkString("/")};"
    val name = split.last.replace("'", "")
    val declObjType = FieldType(declaredType)

    val getInstr: PartialFunction[Instruction, Instruction] = {
      case instr: MethodInvocationInstruction
          if instr.name == name &&
            (instr.declaringClass == declObjType ||
              (declObjType == ObjectType.Object && instr.declaringClass.isArrayType)) => instr
      case instr: INVOKEDYNAMIC => instr
    }

    val calls = method.body.get.collect(getInstr)

    if (calls.size <= number && method.isBridge) {
      computePC(classFile, p, declaredTgt, number, resolveBridgeMethod(classFile, p, method))
    } else if (calls.size > number) {
      Some(calls(number).pc)
    } else {
      None
    }
  }

  // private def parseSignature(rawSig: String): (String, String, MethodDescriptor) = {
  //   val Array(clsAndName, desc) = rawSig.split(":")
  //   val lastDot = clsAndName.lastIndexOf('.')
  //   val className = clsAndName.substring(0, lastDot)
  //   val methodName = clsAndName.substring(lastDot + 1)
  //   val md = MethodDescriptor(desc)
  //   (className, methodName, md)
  // }

    def javaTypeToDescriptor(jtype: String): String = {
    val trimmed = jtype.trim
    if (trimmed.endsWith("[]")) {
      "[" + javaTypeToDescriptor(trimmed.dropRight(2))
    } else {
      val primitives = Map(
        "void" -> "V", "int" -> "I", "float" -> "F", "double" -> "D",
        "long" -> "J", "boolean" -> "Z", "char" -> "C", "short" -> "S", "byte" -> "B"
      )
      primitives.getOrElse(trimmed, s"L${trimmed.replace(".", "/")};")
    }
  }

  def formatSignatureWala(rawSig: String): String = {
    if (!rawSig.contains(":")) return rawSig

    val Array(classPartRaw, rest) = rawSig.split(":", 2)
    val classPart = classPartRaw.trim.replace('.', '/')

    val signatureRegex: Regex = """\s*(\S+)\s+([<>a-zA-Z0-9_$]+)\((.*)\)""".r
    rest.trim match {
      case signatureRegex(returnType, methodName, paramStr) =>
        val rt = javaTypeToDescriptor(returnType)
        val paramTypes = if (paramStr.trim.isEmpty) "" else {
          paramStr.split(",").map(p => javaTypeToDescriptor(p.trim)).mkString("")
        }
        s"$classPart.$methodName:($paramTypes)$rt"

      case _ =>
        println(s"Warning: Unable to parse signature: $rawSig")
        rawSig
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println("Usage: <program.jar> <jdk.jar> <input-callgraph.csv> <output.csv>")
      return
    }

    val inputJar = new File(args(0))
    val jdkPath = new File(args(1))
    val inputCsv = new File(args(2))
    val outputCsv = new File(args(3))

    val project = Project(Array(inputJar, jdkPath), Array.empty[File])

    val writer = new PrintWriter(outputCsv)
    writer.println("method,offset,target")

    for ((line, lineNumber) <- Source.fromFile(inputCsv).getLines().zipWithIndex.drop(1)) {
      val parts = line.split("\t")
      if (parts.length != 4) {
        println(s"[!] Skipping malformed line $lineNumber")
      } else {
        val callerField = parts(1)
        val calleeField = parts(3)

        if (!callerField.contains("/")) {
          println(s"[!] Skipping invalid caller at line $lineNumber: $callerField")
        } else {
          try {

            val callerParts = callerField.split("/")
            if (callerParts.length < 2) {
              println(s"[!] Skipping invalid declared target at line $lineNumber: $callerField")
              return
            }
            val declredTrg = callerParts(1)
            val caller = callerParts(0)

            val methodSig = caller.stripPrefix("<").stripSuffix(">")
            val targetSig = calleeField.stripPrefix("<").stripSuffix(">")


            val methodReformat = formatSignatureWala(methodSig)
            val targetReformat = formatSignatureWala(targetSig)
            val offset = callerParts(2).toInt

            val Array(callerClassMethod, callerSignature)  = methodReformat.split(":")
            val Array(callerClass, callerMethod) = callerClassMethod.split("\\.")

            val descriptor = MethodDescriptor(callerSignature)

            project.classFile(ObjectType(callerClass)) match {
              case Some(cf) =>
                cf.findMethod(callerMethod, descriptor) match {
                  case Some(method) if method.body.isDefined =>
                    computePC(cf, project, declredTrg, offset, method) match {
                      case Some(pc) => writer.println(s"$methodReformat,$pc,$targetReformat")
                      case None => println(s"[!] Could not compute PC at line $lineNumber")
                    }
                  case Some(_) => println(s"[!] Method has no body at line $lineNumber: $methodReformat")
                  case None    => println(s"[!] Method not found at line $lineNumber: $methodReformat")
                }
              case None => println(s"[!] Class not found at line $lineNumber: $callerClass")
            }
          } catch {
            case ex: Throwable => println(s"[!] Error at line $lineNumber: ${ex.getMessage}")
          }
        }
      }
    }

    writer.close()
  }
}

// implicit class RichString(val s: String) extends AnyVal {
//   def rsplit(sep: String, max: Int): Array[String] = {
//     val idx = s.lastIndexOf(sep)
//     if (idx == -1) Array(s) else Array(s.substring(0, idx), s.substring(idx + 1))
//   }
// }
