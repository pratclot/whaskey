import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const ratedItemSchema = z.object({
  name: z.string(),
  category: z.enum(['Whisky', 'Beer & Ale', 'Spirits & Liqueur', 'Food & Scran', 'Other']).default('Whisky'),
  brand_or_distillery: z.string().optional().default(''),
  region_or_style: z.string().optional().default(''),
  abv: z.string().optional().default(''),
  timestamp_seconds: z.number().default(0),
  timestamp_display: z.string().default('00:00'),
  rating_score: z.number().nullable().optional().default(null),
  rating_raw: z.string().default('Unrated'),
  tags: z.array(z.string()).default([]),
  tasting_notes: z.string().default(''),
  verdict_quote: z.string().optional().default(''),
});

const reviews = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/reviews' }),
  schema: z.object({
    video_id: z.string(),
    video_title: z.string(),
    publish_date: z.string(),
    thumbnail: z.string().optional(),
    has_reviews: z.boolean().default(true),
    manually_edited: z.boolean().default(false),
    transcript_source: z.string().optional(),
    episode_summary: z.string().optional().default(''),
    items: z.array(ratedItemSchema).default([]),
  }),
});

export const collections = { reviews };
