# TypeScript/JavaScript

This Lorem Ipsum generator can also be used in TypeScript or JavaScript. You can use npm package or include script file directly in your HTML file.

## npm package

The package is available at: [https://www.npmjs.com/package/lorem-ipsum-nn](https://www.npmjs.com/package/lorem-ipsum-nn).

```shell
npm install lorem-ipsum-nn
```

```typescript
import { LoremIpsum } from 'lorem-ipsum-nn';
```

## Script Tag

Script files can be downloaded from [**Github Releases**](https://github.com/kildom/lorem-ipsum-nn/releases/latest)

  ```html
  <script src="loerm-ipsum.min.js"></script>
  ```

## Usage

```typescript
let generator = LoremIpsum({ /* Options */});
console.log(generator.generate(1000));
```

> *API documentation: Work in progress*
>
> For API details see [lorem-ipsum.ts](./lorem-ipsum.ts).
