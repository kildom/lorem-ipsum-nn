import { promises as fs } from 'fs';
import * as path from 'path';


async function renameJsFile(filePath: string, ext: string) {
    console.log(`${ext}: `, `Renaming: ${filePath}`);

    // Extract directory and filename without extension
    const dir = path.dirname(filePath);
    const baseName = path.basename(filePath, '.js');

    // Create new file paths
    const newFilePath = path.join(dir, `${baseName}.${ext}`);
    const mapFilePath = path.join(dir, `${baseName}.js.map`);
    const newMapFilePath = path.join(dir, `${baseName}.${ext}.map`);

    // Step 2: Rename the JS file
    await fs.rename(filePath, newFilePath);
    console.log(`${ext}: `, `  Renamed JS: ${path.basename(filePath)} → ${path.basename(newFilePath)}`);

    // Step 3: Update sourceMappingURL in the renamed file
    const fileContent = await fs.readFile(newFilePath, 'utf8');
    const oldMapReference = `${baseName}.js.map`;
    const newMapReference = `${baseName}.${ext}.map`;
    const updatedContent = fileContent.replace(
        `//# sourceMappingURL=${oldMapReference}`,
        `//# sourceMappingURL=${newMapReference}`
    );
    if (updatedContent === fileContent) {
        throw new Error(`No sourceMappingURL in ${path.basename(newFilePath)}`);
    }
    await fs.writeFile(newFilePath, updatedContent, 'utf8');
    console.log(`${ext}: `, `  Updated sourceMappingURL in ${path.basename(newFilePath)}`);

    // Step 4: Process map file if it exists
    // Read and parse the map file
    const mapContent = await fs.readFile(mapFilePath, 'utf8');
    const mapData = JSON.parse(mapContent);

    // Update the "file" field in the map data
    mapData.file = `${baseName}.${ext}`;

    // Write the updated map file with new name
    await fs.writeFile(newMapFilePath, JSON.stringify(mapData), 'utf8');
    console.log(`${ext}: `, `  Updated map file: ${path.basename(mapFilePath)} → ${path.basename(newMapFilePath)}`);

    // Remove the old map file
    await fs.unlink(mapFilePath);
}


async function processJsFiles(dir: string, newExtension: string) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            await processJsFiles(fullPath, newExtension);
        } else if (entry.isFile() && entry.name.endsWith('.js')) {
            await renameJsFile(fullPath, newExtension);
        }
    }
}

async function replaceBrowserDecl(filePath: string) {
    console.log(`Replacing browser declaration in: ${filePath}`);

    // Read the file content
    const fileContent = await fs.readFile(filePath, 'utf8');

    // Replace the magic string with the LoremIpsum declaration
    const updatedContent = fileContent.replace('YTgst2yhZeXaCa', '{ LoremIpsum }');

    // Check if replacement was made
    if (updatedContent === fileContent) {
        throw new Error(`No magic string found in ${path.basename(filePath)}`);
    }

    // Write the updated content back to the file
    await fs.writeFile(filePath, updatedContent, 'utf8');
    console.log(`  Replaced magic string in ${path.basename(filePath)}`);

    console.log(`Creating browser declaration for: ${filePath}`);

    // Create .d.ts file path from the original file path
    const dir = path.dirname(filePath);
    const baseName = path.basename(filePath, path.extname(filePath));
    const dtsFilePath = path.join(dir, `${baseName}.d.ts`);

    // Read the source .d.ts content from dist/esm/lorem-ipsum.d.ts
    const sourceDtsPath = 'dist/esm/lorem-ipsum.d.ts';
    const sourceDtsContent = await fs.readFile(sourceDtsPath, 'utf8');

    // Append the global declaration
    const globalDeclaration = '\ndeclare global {\n    var LoremIpsum: typeof LoremIpsum;\n}\n';

    const finalContent = sourceDtsContent + globalDeclaration;

    // Write the new .d.ts file
    await fs.writeFile(dtsFilePath, finalContent, 'utf8');
    console.log(`  Created browser declaration file: ${path.basename(dtsFilePath)}`);
}


processJsFiles('dist/cjs', 'cjs')
processJsFiles('dist/esm', 'esm')
replaceBrowserDecl('dist/browser/lorem-ipsum.js')
replaceBrowserDecl('dist/browser/lorem-ipsum.min.js')

