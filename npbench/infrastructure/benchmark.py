# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import json
import pathlib
import numpy as np

from typing import Any, Dict, Optional


class Benchmark(object):
    """ A class for reading and benchmark information and initializing
    bechmark data. """

    def __init__(self, bname: str):
        """ Reads benchmark information.
        :param bname: The benchmark name.
        """

        self.bname = bname
        self.bdata = dict()

        parent_folder = pathlib.Path(__file__).parent.absolute()
        bench_filename = "{b}.json".format(b=bname)
        bench_path = parent_folder.joinpath("..", "..", "bench_info", bench_filename)
        try:
            with open(bench_path) as json_file:
                self.info = json.load(json_file)["benchmark"]
                # print(self.info)
        except Exception as e:
            print("Benchmark JSON file {b} could not be opened.".format(b=bench_filename))
            raise (e)

    def get_data(self, preset: str = 'L', datatype: Optional[str] = None) -> Dict[str, Any]:
        """ Initializes the benchmark data.
        :param preset: The data-size preset (S, M, L, paper).
        """

        if preset in self.bdata.keys():
            return self.bdata[preset]

        # 1. Create data dictionary
        data = dict()
        # 2. Add parameters to data dictionary
        if preset not in self.info["parameters"].keys():
            raise NotImplementedError("{b} doesn't have a {p} preset.".format(b=self.bname, p=preset))
        parameters = self.info["parameters"][preset]
        for k, v in parameters.items():
            data[k] = v
        if datatype is not None:
            all_datatypes = {"float32": np.float32, "float64": np.float64}
            if datatype not in all_datatypes:
                raise NotImplementedError("Datatype {} is not supported.".format(datatype))
            data["datatype"] = all_datatypes[datatype]
        # 3. Import initialization function
        if "init" in self.info.keys() and self.info["init"]:
            module_filename = "{m}.py".format(m=self.info["module_name"])
            module_pypath = "npbench.benchmarks.{r}.{m}".format(r=self.info["relative_path"].replace('/', '.'),
                                                                m=self.info["module_name"])
            exec_str = "from {m} import {i}".format(m=module_pypath, i=self.info["init"]["func_name"])
            try:
                exec(exec_str, data)
            except Exception as e:
                print("Module Python file {m} could not be opened.".format(m=module_filename))
                raise (e)
            # 4. Execute initialization
            maybe_datatype = ["datatype"] if datatype is not None else []
            init_str = "{oargs} = {i}({iargs})".format(oargs=",".join(self.info["init"]["output_args"]),
                                                       i=self.info["init"]["func_name"],
                                                       iargs=",".join(self.info["init"]["input_args"] + maybe_datatype))
            exec(init_str, data)
            del data[self.info["init"]["func_name"]]

        self.bdata[preset] = data
        return self.bdata[preset]
