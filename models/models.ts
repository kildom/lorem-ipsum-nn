
import { laModel } from "./la";
import { elModel } from "./el";
import { enModel } from "./en";
import { plModel } from "./pl";
import { ukModel } from "./uk";

export const models: { [key: string]: typeof laModel } = {
    la: laModel,
    el: elModel,
    en: enModel,
    pl: plModel,
    uk: ukModel,
};

export { laModel as defaultModel };
