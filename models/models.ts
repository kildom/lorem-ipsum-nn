
import { laModel } from "./la";
import { enModel } from "./en";
import { plModel } from "./pl";

export const models: { [key: string]: typeof laModel } = {
    la: laModel,
    en: enModel,
    pl: plModel,
};

export { laModel as defaultModel };
