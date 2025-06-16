import java.io.{FileWriter, File}

object SerializeRunner {
  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      println("Usage: <algorithm> <inputDir> <outputFile>")
      System.exit(1)
    }

    val algorithm = args(0)
    val inputDir = args(1)
    val outputPath = args(2)


    val options = AdapterOptions.makeJavaOptions("Entrypoint", Array(inputDir), "path/to/JCK", true)

    val writer = new FileWriter(new File(outputPath))

    val duration = OpalJCGAdatper.serializeCG(algorithm, inputDir, writer, options)
    writer.close()

    println(s"Call graph serialized in ${duration / 1_000_000} ms")
  }
}