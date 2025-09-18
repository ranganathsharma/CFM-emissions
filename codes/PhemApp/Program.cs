// This file has the c# code for processing a given folder full of velocity time series to emissions data. 
// Use the command dotnet build -c Release in the powershell to create the .exe file which will lie in the bin folder
// This file will be run through the python file named coupling which iteratively runs the file making and emissions calculation

using System.Globalization;
using PHEMlightdll;

namespace PHEMLightApp
{
    internal class PHEMLight
    {
        private Start calcPHEMlight = new Start();
        private string standard;
        private string engine;
        private string baseDir = @"C:\Users\Administrator\OneDrive - Trinity College Dublin\Admin\PHEMlight\PHEMLight_1.0.5_TCD\PHEMLight_1.0.5_TCD";
        private string filesDir => Path.Combine(baseDir, "Files");
        private string outputDir = @"C:\Users\Administrator\OneDrive - Trinity College Dublin\Academic\PhD\NCFM\HPC\Emissions\Iterative_outputs";

        private string VEHspez => Path.Combine(baseDir, $@"Default Vehicles\V5\PC_{standard}_{engine}.PHEMLight.veh");
        private string FCFull => Path.Combine(baseDir, $@"Default Vehicles\V5\PC_{standard}_{engine}_FC.csv");
        private string EMIFull => Path.Combine(baseDir, $@"Default Vehicles\V5\PC_{standard}_{engine}.csv");
        private Correction DataCor;

        public PHEMLight(string engineType, string standardType)
        {
            engine = engineType;
            standard = standardType;
            DataCor = new Correction(filesDir);
        }

        public void Run()
        {
            string inputDir = @"C:\Users\Administrator\OneDrive - Trinity College Dublin\Academic\PhD\NCFM\HPC\Emissions\Iterative_inputs";
            foreach (var file in Directory.GetFiles(inputDir, "*.csv"))
            {
                List<double> vel = LoadSingleRowCsv(file);
                int len = vel.Count;

                List<double> time = new List<double>();
                List<double> grad = new List<double>();

                for (int i = 0; i < len; i++)
                {
                    time.Add(i*1);      // 1s time steps
                    grad.Add(0.0);      // flat gradient
                }

                List<VehicleResult> _Result = new List<VehicleResult>();
                calcPHEMlight.CALC_Array(new List<string> { VEHspez, FCFull, EMIFull }, time, vel, grad, out _Result, false, DataCor, "c");

                if (_Result == null || _Result.Count == 0)
                {
                    Console.WriteLine($"Error processing {Path.GetFileName(file)}: {calcPHEMlight.Helper.ErrMsg}");
                    continue;
                }

                string outputName = Path.GetFileNameWithoutExtension(file);
                calcPHEMlight.ExportData(Path.Combine(outputDir, outputName + "_Output.csv"), _Result);
                Console.WriteLine($"Processed: {outputName}");
            }
        }

        private List<double> LoadSingleRowCsv(string path)
        {
            List<double> values = new List<double>();
            string content = File.ReadAllText(path);
            string[] parts = content.Split(',');

            foreach (var part in parts)
            {
                if (double.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out double val))
                {
                    values.Add(val);
                }
            }

            return values;
        }
        public static void Main(string[] args)
        {
            string engine = args.Length > 0 ? args[0] : "D";
            string standard = args.Length > 1 ? args[1] : "EU6ab";

            var calc = new PHEMLight(engine, standard);
            calc.Run();
        }

    }
}